#!/bin/bash
# ============================================================
# 外部数据 SVM 训练脚本
# 说明：使用 data/external/ 下的独立数据集训练并评估 SVM
# 用法：bash scripts/train_svm_external.sh
# ============================================================

# 出错时停止
set -e

# 项目根目录（脚本所在目录的上一级）
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

EXPERIMENT_ID="svm_external_001"
CONFIG="config/svm_external.yaml"

echo "=========================================="
echo " 开始时间: $(date)"
echo " 实验 ID:  $EXPERIMENT_ID"
echo " 配置:     $CONFIG"
echo "=========================================="

# Step 1: 训练 SVM（内部自动划分 train/val，test_size=0）
echo ""
echo "[Step 1/2] 训练 SVM ..."
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

# Step 3: 打印关键指标
echo ""
echo "=========================================="
echo " 实验结果汇总"
echo "=========================================="
METRICS_FILE="outputs/${EXPERIMENT_ID}/metrics.json"
EXT_METRICS_FILE="outputs/${EXPERIMENT_ID}/external_test_metrics.json"

if [ -f "$METRICS_FILE" ]; then
    echo "--- 训练/验证集指标 ---"
    python -c "
import json
d = json.load(open('${METRICS_FILE}'))
for split in ['train', 'val']:
    if split in d:
        print(f'  {split}: macro_f1={d[split].get(\"macro_f1\", \"N/A\"):.4f}, balanced_acc={d[split].get(\"balanced_accuracy\", \"N/A\"):.4f}')
"
fi

if [ -f "$EXT_METRICS_FILE" ]; then
    echo "--- 独立测试集指标 ---"
    python -c "
import json
d = json.load(open('${EXT_METRICS_FILE}'))
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
