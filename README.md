# Histopathology Image Classification with ResNet & Grad-CAM

This repository implements a deep learning approach for the classification of lung and colon histopathological images using Transfer Learning. The project emphasizes not only classification accuracy but also **Model Interpretability** through the use of Gradient-weighted Class Activation Mapping (Grad-CAM).

## 🔬 Project Scope

The implementation utilizes the **LC25000 Dataset**, covering 5 distinct histopathological tissue classes:
* `colon_aca` / `colon_n`: Adenocarcinoma vs. Benign colonic tissue.
* `lung_aca` / `lung_scc` / `lung_n`: Adenocarcinoma, Squamous Cell Carcinoma, and Benign lung tissue.

## 🛠️ Technical Approach

* **Transfer Learning**: Employs a pre-trained **ResNet18** backbone fine-tuned on histopathological textures. The architecture leverages ImageNet-derived features while adapting the final layers to specialized medical imagery.
* **Grad-CAM Visualization**: Integrates `torchcam` to generate heatmap overlays. This feature highlights the specific regions within the tissue slides that most influence the model's classification, providing a layer of transparency to the "black box" nature of the neural network.
* **Data Augmentation**: Incorporates a robust PyTorch-based pipeline with `RandomResizedCrop` and `HorizontalFlip` to improve model generalization across different scanning conditions.

## 📁 Directory Layout

```text
Histopathology-Classification/
├── src/                     # Model definitions and Grad-CAM integration
├── notebooks/               # Comprehensive Classification & Visualization Demo
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

# 基于 ResNet 与 Grad-CAM 的病理切片图像分类

本项目利用迁移学习实现肺部和结肠病理组织学图像的深度学习分类。除了关注分类准确率外，本项目还重点通过梯度加权类激活映射（Grad-CAM）技术实现**模型的可解释性**。

## 🔬 项目范围

该实现采用了 **LC25000 数据集**，涵盖了 5 种不同的病理组织类别：
* `colon_aca` / `colon_n`：结肠腺癌 vs. 良性结肠组织。
* `lung_aca` / `lung_scc` / `lung_n`：肺腺癌、肺鳞状细胞癌及良性肺组织。

## 🛠️ 技术方法

* **迁移学习**：采用预训练的 **ResNet18** 作为骨干网络，并在病理图像纹理上进行微调。该架构利用了从 ImageNet 学习到的特征，同时将最终层适配于专门的医学图像。
* **Grad-CAM 可视化**：集成了 `torchcam` 以生成热力图叠加。此功能突出了组织切片中对模型分类影响最大的特定区域，为神经网络的“黑盒”属性提供了一层透明度。
* **数据增强**：整合了基于 PyTorch 的稳健流水线，包含 `RandomResizedCrop` 和 `HorizontalFlip` 等操作，以提高模型在不同扫描条件下的泛化能力。

## 📁 目录结构

```text
Histopathology-Classification/
├── src/                     # 模型定义与 Grad-CAM 集成
├── notebooks/               # 完整的分类与可视化演示
├── requirements.txt         # 所需 Python 依赖包
└── README.md                # 项目文档
```
