#!/bin/bash
# ============================================================
# 外部数据 SVM 最优配置训练脚本
# 说明：使用 data/external/ 下的独立数据集，采用
#       svm_exp004 验证的最优特征/预处理/PCA 配置
# 用法：bash scripts/train_svm_external_best.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

EXPERIMENT_ID="svm_external_best"
CONFIG="config/svm_external.yaml"

echo "=========================================="
echo " 开始时间: $(date)"
echo " 实验 ID:  $EXPERIMENT_ID"
echo " 配置:     $CONFIG"
echo " 特点: PCA+CLAHE+颜色归一化+11组特征"
echo "=========================================="

# Step 1: 训练
echo ""
echo "[Step 1/2] 训练 SVM（最优配置）..."
python src/train.py \
    --config "$CONFIG" \
    --experiment_id "$EXPERIMENT_ID"

if [ $? -ne 0 ]; then
    echo "错误：训练失败！"
    exit 1
fi
echo "训练完成。"

# Step 2: 在独立测试集上评估
echo ""
echo "[Step 2/2] 在独立测试集上评估 ..."
python scripts/evaluate_external.py \
    --config "$CONFIG" \
    --experiment_id "$EXPERIMENT_ID" \
    --test_csv data/external/test/label.csv \
    --test_image_dir data/external/test/image \
    --test_mask_dir data/external/test/mask

if [ $? -ne 0 ]; then
    echo "错误：评估失败！"
    exit 1
fi
echo "评估完成。"

# Step 3: 汇总结果
echo ""
echo "=========================================="
echo " 实验结果汇总"
echo "=========================================="
EXT_FILE="outputs/${EXPERIMENT_ID}/external_test_metrics.json"
if [ -f "$EXT_FILE" ]; then
    echo "--- 独立测试集指标 ---"
    python -c "
import json
d = json.load(open('${EXT_FILE}'))
for k, v in d.items():
    if k != 'classification_report':
        print(f'  {k}: {v:.4f}')
"
fi

echo ""
echo "=========================================="
echo " 完成时间: $(date)"
echo " 输出目录: outputs/${EXPERIMENT_ID}/"
echo "=========================================="
