import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class HistopathologyResNet(nn.Module):
    """
    病理切片图像分类模型 (基于 ResNet18)。
    集成官方/本地预训练权重加载，并提供获取特征图的接口供 Grad-CAM 使用。
    """
    
    def __init__(self, num_classes: int = 5, local_weights_path: str = None):
        """
        参数:
            num_classes: 分类数目，默认为 5 类。
            local_weights_path: 本地预训练权重 (.pth) 路径。若不为空则优先读取本地。
        """
        super().__init__()
        
        # 加载骨干网络
        if local_weights_path and os.path.exists(local_weights_path):
            self.backbone = models.resnet18(weights=None)
            self._load_local_weights(local_weights_path)
        else:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
        # 重构分类头
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # 对新初始化的全连接层进行 Kaiming 初始化
        nn.init.kaiming_normal_(self.backbone.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.backbone.fc.bias is not None:
            nn.init.zeros_(self.backbone.fc.bias)

    def _load_local_weights(self, weights_path: str):
        """加载本地权重，自动跳过尺寸不匹配的 fc 层。"""
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model_dict = self.backbone.state_dict()
        
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and not k.startswith('fc')
        }
        
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_cam_layer(self) -> nn.Module:
        """返回 layer4 的最后一个残差块，用于提取 CAM 特征。"""
        return self.backbone.layer4[-1]


class GradCAM:
    """Grad-CAM 实现。通过 Hook 机制提取特征图和梯度。"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        # 注册 Hook
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image: torch.Tensor, target_class: int = None):
        """
        计算输入图像在目标类别下的 Grad-CAM 热力图。
        
        参数:
            input_image: 输入图像，形状为 (1, C, H, W)
            target_class: 目标类别索引。若为 None，则使用预测概率最大的类
            
        返回值:
            cam: 归一化后的热力图，尺寸与输入图像一致
            target_class: 实际使用的类别索引
        """
        self.model.zero_grad()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 针对目标类别进行反向传播
        output[0, target_class].backward()

        # 对梯度进行全局平均池化得到通道权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 权重与特征图线性组合，并经过 ReLU 激活
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        # 缩放至原图尺寸并进行 Min-Max 归一化
        _, _, h, w = input_image.shape
        cam = cv2.resize(cam, (w, h))
        
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, target_class

    def remove_hooks(self):
        """移除 Hook 以防止内存泄漏。"""
        self._forward_hook.remove()
        self._backward_hook.remove()
