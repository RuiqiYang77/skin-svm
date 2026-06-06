# 方案A 模型预测
python src/predict.py \
    --config config/svm_glcm_best.yaml \
    --experiment_id planA_exp001 \
    --image_path data/image/1.jpg \
    --mask_path data/mask/mask_1.jpg

# 方案B 模型预测（注释掉，使用时取消注释并运行方案B训练）
# python src/predict.py \
#     --config config/svm_v14b_L4.yaml \
#     --experiment_id planB_exp001 \
#     --image_path data/image/1.jpg \
#     --mask_path data/mask/mask_1.jpg
