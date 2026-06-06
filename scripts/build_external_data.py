"""
从 data_expand (HAM10000) 中挑选外部数据，构建 data/external/ 结构。

挑选规则：
  - 每个类别 (nv, mel, vasc) 挑选 100 张训练 + 50 张测试（vasc 不足则尽可能多）
  - 对训练集做水平翻转增强（_aug1）
  - 输出到 data/external/train/ 和 data/external/test/
"""

import random
import shutil
from pathlib import Path

import pandas as pd

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_IMAGE_DIR = PROJECT_ROOT / "data_expand" / "image"
SRC_MASK_DIR = PROJECT_ROOT / "data_expand" / "mask"
SRC_LABEL_CSV = PROJECT_ROOT / "data_expand" / "label.csv"

DST_BASE = PROJECT_ROOT / "data" / "external"
DST_TRAIN_IMAGE = DST_BASE / "train" / "image"
DST_TRAIN_MASK = DST_BASE / "train" / "mask"
DST_TEST_IMAGE = DST_BASE / "test" / "image"
DST_TEST_MASK = DST_BASE / "test" / "mask"

N_TRAIN_PER_CLASS = 100
N_TEST_PER_CLASS = 50
RANDOM_STATE = 42

# 确保输出目录存在
for d in [DST_TRAIN_IMAGE, DST_TRAIN_MASK, DST_TEST_IMAGE, DST_TEST_MASK]:
    d.mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    """拷贝文件，如目标已存在则跳过。"""
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True


def main():
    random.seed(RANDOM_STATE)

    # 1. 读取标签
    df = pd.read_csv(SRC_LABEL_CSV)
    print(f"data_expand 总图像数: {len(df)}")

    train_rows = []
    test_rows = []

    for cls in ["nv", "mel", "vasc"]:
        cls_df = df[df["dx"] == cls]
        ids = cls_df["image_id"].tolist()
        random.shuffle(ids)

        n_available = len(ids)
        n_test = min(N_TEST_PER_CLASS, n_available // 2)  # 至少留一半给训练
        n_train = min(N_TRAIN_PER_CLASS, n_available - n_test)

        if n_train < 1:
            print(f"  ⚠ {cls}: 仅 {n_available} 张，不足以构建训练+测试，跳过")
            continue

        test_ids = ids[:n_test]
        train_ids = ids[n_test : n_test + n_train]

        print(f"\n{'='*50}")
        print(f"类别 {cls}: 共 {n_available} 张")
        print(f"  → 训练: {len(train_ids)} 张 (ID: {train_ids[0]} ~ {train_ids[-1]})")
        print(f"  → 测试: {len(test_ids)} 张 (ID: {test_ids[0]} ~ {test_ids[-1]})")

        # 拷贝测试集
        for img_id in test_ids:
            src_img = SRC_IMAGE_DIR / f"{img_id}.jpg"
            src_mask = SRC_MASK_DIR / f"mask_{img_id}.jpg"
            dst_img = DST_TEST_IMAGE / f"{img_id}.jpg"
            dst_mask = DST_TEST_MASK / f"mask_{img_id}.jpg"
            copy_file(src_img, dst_img)
            copy_file(src_mask, dst_mask)
            test_rows.append({"image_id": str(img_id), "dx": cls})

        # 拷贝训练集（原始 + 增强）
        for img_id in train_ids:
            # --- 原始 ---
            src_img = SRC_IMAGE_DIR / f"{img_id}.jpg"
            src_mask = SRC_MASK_DIR / f"mask_{img_id}.jpg"
            dst_img = DST_TRAIN_IMAGE / f"{img_id}.jpg"
            dst_mask = DST_TRAIN_MASK / f"mask_{img_id}.jpg"
            copy_file(src_img, dst_img)
            copy_file(src_mask, dst_mask)
            train_rows.append({"image_id": str(img_id), "dx": cls})

            # --- 水平翻转增强 (_aug1) ---
            aug_img_id = f"{img_id}_aug1"
            dst_aug_img = DST_TRAIN_IMAGE / f"{aug_img_id}.jpg"
            dst_aug_mask = DST_TRAIN_MASK / f"mask_{aug_img_id}.jpg"

            # 图像翻转（用 PIL 加载并翻转）
            from PIL import Image
            img = Image.open(src_img)
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_img.save(dst_aug_img, quality=95)

            mask = Image.open(src_mask).convert("L")
            flipped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_mask.save(dst_aug_mask, quality=95)

            train_rows.append({"image_id": aug_img_id, "dx": cls})

        print(f"  (增强后训练集共 {len(train_rows)} 条)")

    # 2. 保存 label.csv
    train_df = pd.DataFrame(train_rows)
    train_csv = DST_BASE / "train" / "label.csv"
    train_df.to_csv(train_csv, index=False)
    print(f"\n✅ 训练标签已保存: {train_csv} ({len(train_df)} 条)")

    test_df = pd.DataFrame(test_rows)
    test_csv = DST_BASE / "test" / "label.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"✅ 测试标签已保存: {test_csv} ({len(test_df)} 条)")

    # 3. 打印摘要
    print(f"\n{'='*50}")
    print("生成结构:")
    print(f"  data/external/train/image/  → {len(list(DST_TRAIN_IMAGE.glob('*.jpg')))} 张")
    print(f"  data/external/train/mask/   → {len(list(DST_TRAIN_MASK.glob('*.jpg')))} 张")
    print(f"  data/external/test/image/   → {len(list(DST_TEST_IMAGE.glob('*.jpg')))} 张")
    print(f"  data/external/test/mask/    → {len(list(DST_TEST_MASK.glob('*.jpg')))} 张")
    print(f"\n训练集按类别分布:")
    print(train_df["dx"].value_counts().to_string())
    print(f"\n测试集按类别分布:")
    print(test_df["dx"].value_counts().to_string())


if __name__ == "__main__":
    main()
