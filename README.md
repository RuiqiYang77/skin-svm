# DIP Skin Lesion Classification

本项目实现一个基于传统图像处理特征和 SVM 的皮肤病变三分类 pipeline。数据包含 200 张原始图像和 400 张增强图像，标签类别为 `nv`、`mel`、`vasc`。

当前主线方法：

```text
lesion mask 预处理 -> 颜色/纹理/形状特征提取 -> 标准化 -> RBF-SVM -> 分类与增强鲁棒性评估
```

## 目录结构

```text
config/
  svm.yaml

data/
  image/
  mask/
  label.csv

src/
  dataloader/
    dataset.py
    preprocessing.py
    features.py
    split.py
  model/
    svm.py
  timm_baseline/
    data.py
    engine.py
    model.py
  utils/
    config.py
    evaluation.py
    io.py
  train.py
  predict.py
  train_timm.py
  predict_timm.py

outputs/
  {experiment_id}/
  timm_{experiment_id}/
```

## 环境配置

建议使用 conda 建独立环境：

```text
conda create -n dip-skin-svm python=3.10
conda activate dip-skin-svm
pip install -r requirement.txt
```

后续如果新增依赖，需要同步更新 `requirement.txt`。

## 训练

```text
python src/train.py --config config/svm.yaml --experiment_id svm_exp001
```

训练输出会保存到：

```text
outputs/svm_exp001/
  config.yaml
  features.csv
  split.csv
  model.joblib
  metrics.json
  predictions.csv
  robustness_detail.csv
  confusion_matrix.png
```

`experiment_id` 必须手动指定，避免多人实验结果互相覆盖。

## 预测

```text
python src/predict.py --config config/svm.yaml --experiment_id svm_exp001 --image_path data/image/1.jpg --mask_path data/mask/mask_1.jpg
```

预测脚本会读取：

```text
outputs/{experiment_id}/model.joblib
```

并输出单张图像的预测类别和各类别概率。

## 迁移学习基线

这部分是一个独立的 timm 深度学习基线，用来和 SVM 主线做对比。

目的：

- 复用当前项目的 `data/label.csv`、`data/image/` 和 `data/mask/`
- 用预训练 timm 模型做端到端微调
- 保留和经典算法类似的输出结构，方便比较
- 和 SVM 主线完全分开，不共享输出目录或中间文件

你可以把它理解成“同一份数据，不同方法”的对照组，适合写报告、做消融和比较模型路线。

训练命令：

```text
python src/train_timm.py --config config/timm_baseline.yaml --experiment_id exp001
```

这条命令适合只跑一个默认模型。默认模型名写在 [config/timm_baseline.yaml](config/timm_baseline.yaml) 里，当前是 `efficientnet_b0`。

如果你只想快速验证这条基线是否能跑通，先用这条单模型命令即可。

如果要一次跑 4 个 timm 模型做对比，可以直接用批量脚本：

```text
python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id exp001
```

默认会依次跑这 4 个模型：

```text
efficientnet_b0
resnet18
vit_base_patch16_224
samvit_base_patch16
```

每个模型都会有自己独立的输出目录，最终汇总会写到：

```text
outputs/timm_exp001/suite_summary.csv
```

如果你只想跑其中一部分，也可以显式传入模型名，例如：

```text
python src/train_timm_suite.py --config config/timm_baseline.yaml --experiment_id exp001 --models efficientnet_b0 resnet18
```

这会跳过其余模型，只跑你指定的子集。批量脚本会重用同一个训练流程，只是把 `model_name` 替换成你给定的架构名，所以所有模型的指标和输出格式是一致的，方便横向比较。

预测命令：

```text
python src/predict_timm.py --config config/timm_baseline.yaml --experiment_id exp001 --image_path data/image/1.jpg --mask_path data/mask/mask_1.jpg
```

输出会保存在：

```text
outputs/timm_exp001/
  config.yaml
  metadata.csv
  split.csv
  history.csv
  model_best.pth
  metrics.json
  predictions.csv
  robustness_detail.csv
  classification_report.txt
  confusion_matrix.png
  single_prediction.csv
```

其中：

- `history.csv` 记录每个 epoch 的训练和验证变化
- `metrics.json` 保存 train / val / test 的汇总指标
- `predictions.csv` 保存每个样本的逐行预测结果
- `robustness_detail.csv` 保存增强组的一致性明细
- `classification_report.txt` 和 `confusion_matrix.png` 是 test 集的最终汇报结果

默认配置文件是：

```text
config/timm_baseline.yaml
```

里面保留了和经典算法类似的训练/验证/测试拆分、早停、混淆矩阵、分类报告和鲁棒性统计，但输出目录和脚本名称都和 SVM 主线区分开了，方便做对比分析。

## 配置

主配置文件是：

```text
config/svm.yaml
```

第一行通过 `model: svm` 指定当前模型。SVM 相关参数都放在 `svm:` 下，包括数据划分、特征开关、预处理、PCA 和网格搜索参数。

评估指标不需要写入 YAML。训练脚本默认报告：

```text
accuracy
balanced_accuracy
macro_precision
macro_recall
macro_f1
classification_report
confusion_matrix
augmentation robustness
```

## 协作规范

多人开发时按目录分工：

1. 数据读取、预处理、特征提取只改 `src/dataloader/`。
2. SVM 模型逻辑只改 `src/model/svm.py`。
3. 训练入口只改 `src/train.py`。
4. 预测入口只改 `src/predict.py`。
5. 配置统一改 `config/svm.yaml`。
6. 依赖统一改 `requirement.txt`。
7. 实验结果统一放到 `outputs/{experiment_id}/`。

## 数据划分原则

增强图像不能随机分散到 train/test。比如：

```text
1.jpg
1_aug1.jpg
1_aug2.jpg
```

这三张图必须属于同一个 split。代码中使用 `base_id` 做 grouped split，避免增强图像造成数据泄漏。

## 当前 SVM 方法说明

这一版先只做传统机器学习主线：使用已有的 lesion mask 提取病灶区域特征，然后用 SVM 做三分类。特征提取代码集中在 `src/dataloader/features.py`，模型代码集中在 `src/model/svm.py`。

当前使用的特征分为三类。

第一类是颜色特征。代码会先用 mask 取出病灶区域像素，然后分别在 RGB、HSV、Lab 三个颜色空间中计算每个通道的 mean、std、skewness。此外，还会计算 HSV 三个通道的直方图、暗色像素比例、Hue 标准差和平均饱和度。这一部分主要用于描述病灶颜色深浅、颜色分布和颜色不均匀程度。

第二类是纹理特征。代码会把图像转成灰度图，并在 mask 的 bounding box 附近计算 LBP histogram 和 GLCM 特征。GLCM 当前包含 contrast、dissimilarity、homogeneity、energy、correlation、ASM，并对多个距离和角度的结果取 mean 和 std。这一部分主要用于描述病灶内部纹理复杂度和灰度变化。

第三类是形状特征。代码直接基于二值 mask 计算 area ratio、perimeter、circularity、eccentricity、major axis length、minor axis length、solidity、extent、bbox aspect ratio、border irregularity、compactness、convex area ratio，以及水平和垂直方向的 asymmetry。这一部分对应皮肤病变分析中常见的面积、边界不规则性和不对称性信息。

训练时，`src/train.py` 会先读取 `config/svm.yaml`，再读取 `data/label.csv` 生成 metadata。metadata 中会为每个样本生成 `image_path`、`mask_path`、`base_id`、`is_augmented` 和 `augmentation_id`。随后代码会提取所有样本的特征，并保存为：

```text
outputs/{experiment_id}/features.csv
```

数据划分使用 `base_id` 分组，保证原图和对应增强图不会被分到不同 split。划分结果保存为：

```text
outputs/{experiment_id}/split.csv
```

SVM 的流程是：

```text
StandardScaler -> PCA(optional) -> SVC(kernel="rbf", class_weight="balanced")
```

当前配置里 PCA 默认关闭，SVM 默认使用 RBF kernel，并设置 `class_weight: balanced` 来缓解 `vasc` 样本较少的问题。如果 `grid_search.enabled: true`，代码会对 `C` 和 `gamma` 做网格搜索，默认评分指标是 `f1_macro`。为了避免部分机器上多进程报错，当前 `n_jobs` 默认设为 1。

当前数据划分、交叉验证和 SVM 概率校准都使用 `random_state: 42`。如果没有修改代码和配置，同一份 `features.csv` 重复训练应得到一致结果。注意：同一个 `experiment_id` 会覆盖 `outputs/{experiment_id}/` 下的结果；如果特征提取代码被改过，重新运行训练会重新生成 `features.csv`，结果可能和旧缓存特征对应的实验不同。

训练完成后会在 train、val、test 三个 split 上都做预测和评估。默认报告 accuracy、balanced accuracy、macro precision、macro recall、macro F1、classification report 和 confusion matrix。额外还会在 test set 上统计增强鲁棒性，也就是同一个 `base_id` 下原图和增强图预测是否一致。最终主要结果保存在：

```text
outputs/{experiment_id}/metrics.json
outputs/{experiment_id}/predictions.csv
outputs/{experiment_id}/robustness_detail.csv
outputs/{experiment_id}/confusion_matrix.png
outputs/{experiment_id}/model.joblib
```

如果后续同学要改特征，只需要优先改 `src/dataloader/features.py` 和 `config/svm.yaml` 中的 feature 开关；如果要改 SVM 参数，只改 `config/svm.yaml` 和必要时的 `src/model/svm.py`。不要把特征提取逻辑写进模型文件里。
