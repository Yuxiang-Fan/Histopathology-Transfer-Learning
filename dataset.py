import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Optional, Callable

class HistopathologyDataset(Dataset):
    """
    病理切片图像数据集。
    动态读取指定目录下的子文件夹作为类别标签，支持自动推断。
    """
    
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        参数:
            data_dir: 数据集根目录，需包含以类别命名的子文件夹。
            transform: 数据预处理/增强流水线。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 扫描目录获取类别，按字母顺序排序保证标签映射的稳定性
        self.classes = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        
        # 遍历读取所有图像路径
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    self.images.append(os.path.join(class_dir, file_name))
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 统一转换为 RGB 格式，防止读取单通道灰度图时导致 Tensor 维度报错
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class TransformSubset(Dataset):
    """
    带独立 Transform 的子集包装器。
    用于解决 PyTorch 原生 Subset 共享底层 Dataset Transform 的引用污染问题。
    """
    
    def __init__(self, subset: Subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 从底层基础数据集中提取 PIL Image 和 Label
        x, y = self.subset[idx]
        
        # 独立应用本子集的 Transform
        if self.transform:
            x = self.transform(x)
        return x, y


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """获取训练集和验证/测试集的图像预处理与增强操作。"""
    
    # ImageNet 统计均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def build_dataloaders(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    构建并划分数据集，生成对应的 DataLoader。
    
    参数:
        data_dir: 数据集根目录路径。
        batch_size: 批次大小。
        num_workers: 异步加载数据的子进程数。
        split_ratios: train, val, test 的划分比例。
        
    返回值:
        train_loader, val_loader, test_loader, classes (类别名称列表)
    """
    # 实例化无 Transform 的基础全量数据集
    base_dataset = HistopathologyDataset(data_dir=data_dir, transform=None)
    
    train_trans, val_trans = get_transforms()
    
    total_size = len(base_dataset)
    train_size = int(total_size * split_ratios[0])
    val_size = int(total_size * split_ratios[1])
    test_size = total_size - train_size - val_size
    
    # 随机打乱全局索引
    indices = torch.randperm(total_size).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 采用安全包装类规避引用覆盖
    train_dataset = TransformSubset(Subset(base_dataset, train_indices), transform=train_trans)
    val_dataset = TransformSubset(Subset(base_dataset, val_indices), transform=val_trans)
    test_dataset = TransformSubset(Subset(base_dataset, test_indices), transform=val_trans)
    
    # 开启 pin_memory=True 以加速 CPU 内存到 GPU 显存的传输
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, base_dataset.classes
