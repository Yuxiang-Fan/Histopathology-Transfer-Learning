# Histopathology Image Classification with ResNet & Custom Grad-CAM
# 基于 ResNet 与自定义 Grad-CAM 的病理切片图像分类


---


### Project Overview
This repository implements a high-performance deep learning framework for the classification of lung and colon histopathological images. The project emphasizes both classification accuracy and **Model Interpretability**, featuring a custom-built Grad-CAM implementation and a robust data pipeline that ensures training/testing isolation.

### Technical Highlights
* **OOP-Based Model Design**: The **ResNet18** architecture is encapsulated in an Object-Oriented pipeline, supporting flexible initialization from local weights (`.pth`) or PyTorch Hub.
* **Custom Grad-CAM (Hook-Based)**: Instead of using third-party libraries, this project implements Grad-CAM from scratch using PyTorch's **Forward and Backward Hooks**. This allows for precise extraction of feature maps and gradients for visual explanation.
* **Robust Data Pipeline**: Features a custom `TransformSubset` wrapper to resolve the native PyTorch **Subset transform pollution** issue, ensuring that data augmentations for training do not leak into the validation or testing phases.
* **Automated Artifact Generation**: A unified `main.py` script automates the entire process, calculating comprehensive metrics (Accuracy, Precision, Recall, F1, and Micro-AUC) and generating ROC curves, Confusion Matrices, and Heatmaps.

### ⚠️ Dataset Access Note
Due to storage constraints and licensing, **the dataset is NOT provided in this repository**. Users are required to independently download the **LC25000 Dataset** (containing 25,000 images of lung and colon tissues) from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) or other official sources.

### Usage
1. Download the LC25000 dataset and organize it into the `data/` directory.
2. (Optional) Place pre-trained weights in `pretrained_model/` for offline initialization.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute the pipeline:
   ```bash
   python main.py
   ```

---


### 项目简介
本项目利用迁移学习实现了一套高性能的肺部和结肠病理组织切片图像分类框架。项目在保证分类准确率的同时，重点关注**模型的可解释性**。通过自定义实现的 Grad-CAM 以及稳健的数据流水线，确保了诊断过程的透明度和数据实验的严谨性。

### 技术亮点
* **基于 OOP 的模型设计**：将 **ResNet18** 架构封装在面向对象的流水线中，支持从本地权重（`.pth`）或 PyTorch Hub 灵活初始化模型。
* **自定义 Grad-CAM（基于 Hook）**：不依赖第三方黑盒库，而是直接基于 PyTorch 的**前向和后向 Hook 机制**从底层开发了 Grad-CAM 模块，能够精确提取特征图和梯度，生成可视化的解释性热力图。
* **稳健的数据流水线**：实现了自定义的 `TransformSubset` 包装类，从根本上解决了原生 PyTorch **Subset 变换污染**（引用干扰）问题，确保训练集的数据增强不会污染验证集或测试集。
* **自动化产物生成**：通过统一的 `main.py` 脚本自动处理全流程，计算包括准确率、精确率、召回率、F1 及 Micro-AUC 在内的多维指标，并自动生成 ROC 曲线、混淆矩阵和 CAM 热力图叠加。

### ⚠️ 数据集获取说明
受限于存储空间及授权协议，**本仓库不提供数据集**。用户需自行从 [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) 或其他官方渠道下载 **LC25000 数据集**（包含 25,000 张肺部和结肠组织图像）。

### 使用说明
1. 下载 LC25000 数据集并将其放入 `data/` 目录。
2. （可选）将预训练权重放入 `pretrained_model/` 文件夹以便离线加载。
3. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```
4. 执行全流程分析：
   ```bash
   python main.py
   ```

---

## 📁 Repository Structure / 仓库结构

```text
.
├── model.py            # ResNet18 architecture & Hook-based Grad-CAM
├── dataset.py          # Data loaders & TransformSubset implementation
├── main.py             # Training, evaluation, and visualization entry
├── pretrained_model/   # Directory for resnet18.pth
├── outputs/            # Generated ROC, CM, and CAM images
└── README.md           # Project documentation
```
