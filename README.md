# Histopathology Image Classification with Transfer Learning & Grad-CAM

A comprehensive deep learning pipeline for classifying lung and colon histopathological images. This project demonstrates the application of Transfer Learning (using ResNet architectures) for high-accuracy medical image classification, coupled with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual interpretability for the model's clinical decisions.

## 🔬 Dataset Overview

This project utilizes the **LC25000 Dataset**, which contains histopathological images spanning 5 distinct tissue classes:
* `colon_aca`: Colon Adenocarcinoma
* `colon_n`: Benign Colonic Tissue
* `lung_aca`: Lung Adenocarcinoma
* `lung_n`: Benign Lung Tissue
* `lung_scc`: Lung Squamous Cell Carcinoma

*Note: Due to size constraints, the dataset is not included in this repository. Please see the Quick Start section for setup instructions.*

## 🛠️ Tech Stack & Architecture

* **Framework**: PyTorch & Torchvision
* **Methodology**: Transfer Learning (ResNet18 pre-trained on ImageNet)
* **Interpretability**: `torchcam` for generating heatmap overlays
* **Data Processing**: Dynamic 80/20 train-validation splitting with runtime data augmentation (RandomResizedCrop, RandomHorizontalFlip).

## 🚀 Quick Start

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
