import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, train_ratio=0.8):
    """加载图像文件夹并动态划分为训练集和验证集"""
    # 统一的图像预处理（由于动态划分，这里使用通用的 Resize 和标准化）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes

    # 计算训练集和验证集的长度
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    # 随机划分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_names