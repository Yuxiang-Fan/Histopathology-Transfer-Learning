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
