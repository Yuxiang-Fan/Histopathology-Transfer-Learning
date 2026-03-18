import torch
import torch.nn as nn
from torchvision import models

def build_resnet(num_classes=5):
    """构建 ResNet18 模型并加载 PyTorch 官方预训练权重"""
    # 开启 pretrained=True，PyTorch 会自动下载官方权重并缓存
    print("Loading pretrained ResNet18 weights from PyTorch hub...")
    model = models.resnet18(pretrained=True)
    
    # 修改最后的全连接层以适应特定的分类任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model