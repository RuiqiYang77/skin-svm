# DIP Skin Lesion Classification

基于传统图像处理特征和 SVM 的皮肤病变三分类（nv / mel / vasc）。数据包含 200 张原始图像和 400 张增强图像。

经过 60+ 次系统消融实验，最终得到两个最优方案。

---

## 方案 A — Macro F1 最优 (F1=0.8744, vasc=1.000)

```bash
python src/train.py --config config/svm_glcm_best.yaml --features_module src.dataloader.features_glcm_levels --experiment_id planA_exp001
```

**Pipeline**: γ=1.06 → CLAHE(kernel=7) → 全局SoG(p=6) → GLCM levels=4 → PCA(0.95) → RBF-SVM

| 指标 | 值 |
|---|---|
| Accuracy | 0.842 |
| Balanced Accuracy | 0.880 |
| Macro F1 | **0.874** |
| mel P/R/F1 | 0.735 / 0.857 / 0.791 |
| nv P/R/F1 | 0.887 / 0.783 / 0.832 |
| vasc P/R/F1 | **1.000 / 1.000 / 1.000** |

---

## 方案 B — Clinical Safe 最优 (mel Recall=0.881, BalAcc=0.888)

```bash
python src/train.py --config config/svm_v14b_L4.yaml --features_module src.dataloader.features_glcm_v14b --experiment_id planB_exp001
```

**Pipeline**: γ=1.08 → CLAHE(kernel=8) → skin-only SoG(p=6) → GLCM levels=4 → PCA(0.95) → RBF-SVM

| 指标 | 值 |
|---|---|
| Accuracy | **0.850** |
| Balanced Accuracy | **0.888** |
| Macro F1 | 0.870 |
| mel P/R/F1 | 0.771 / **0.881** / 0.822 |
| nv P/R/F1 | **0.904** / 0.783 / **0.839** |
| vasc Recall | 1.000 |

---

## 目录结构

```text
config/
  svm_glcm_best.yaml         # 方案A 配置 (γ=1.06, k=7, GLCM levels=4)
  svm_v14b_L4.yaml           # 方案B 配置 (γ=1.08, k=8, skin-only SoG, GLCM levels=4)

src/
  dataloader/
    preprocessing.py          # 标准 v1 预处理 (CLAHE + 全局SoG)
    preprocessing_gamma.py    # γ可调预处理 (方案A用)
    preprocessing_v14b.py     # skin-only SoG 预处理 (方案B用)
    features.py               # 基线特征提取 (347维)
    features_glcm_levels.py   # GLCM levels可调特征 (方案A用)
    features_glcm_v14b.py     # GLCM levels + skin-only SoG (方案B用)
    dataset.py                # 元数据构建
    split.py                  # 分组数据划分
  model/svm.py                # RBF-SVM模型
  utils/
    config.py / evaluation.py / io.py
  train.py                    # 训练入口
  predict.py                  # 单张预测入口

doc/
  optimization_report.md      # 完整优化报告 (60+实验记录)
```

## 环境配置

```bash
conda create -n dip-skin-svm python=3.10
conda activate dip-skin-svm
pip install -r requirement.txt
```

## 数据格式

```text
data/
  image/          # 皮肤镜图像 .jpg
  mask/           # 病灶 mask .jpg (命名: mask_{image_id}.jpg)
  label.csv       # image_id,dx 两列，dx取值为 nv/mel/vasc
```

增强图像以 `_aug1`、`_aug2` 后缀命名，代码通过 base_id 分组保证增强图与原图在同一 split 中。

## 方案选择建议

- 以综合评分排名为目标 → **方案A** (F1=0.8744)
- 以临床安全性为首要（宁可误报不可漏诊）→ **方案B** (mel Recall=0.881, BalAcc=0.888)
