import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# 从本地模块导入模型与数据构建流水线
from model import HistopathologyResNet, GradCAM
from dataset import build_dataloaders

def train_and_evaluate(model: nn.Module, train_loader, val_loader, criterion, optimizer, num_epochs: int, device: torch.device, output_dir: str):
    """模型训练与验证主循环。"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_resnet18.pth')

    for epoch in range(num_epochs):
        # ---------------- 训练阶段 ----------------
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_acc = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # ---------------- 模型保存 ----------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] 验证集损失下降，已保存最佳模型至: {best_model_path}")

    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'loss_curve.svg'), format='svg')
    plt.close()

    return best_model_path


def test_model(model: nn.Module, test_loader, device: torch.device, num_classes: int, class_names: list, output_dir: str):
    """测试集评估与指标计算。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算分类指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print("\n" + "="*40)
    print("测试集最终评估指标:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Macro)")
    print(f"Recall:    {recall:.4f} (Macro)")
    print(f"F1 Score:  {f1:.4f} (Macro)")
    print("="*40 + "\n")

    # ---------------- 绘制混淆矩阵 ----------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.svg'), format='svg')
    plt.close()

    # ---------------- 绘制 ROC 曲线 ----------------
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    
    # 逐类计算 ROC 与 AUC
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    # 计算 Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linestyle=':', linewidth=3, color='black')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.svg'), format='svg')
    plt.close()


def run_gradcam(model: nn.Module, test_loader, device: torch.device, output_dir: str, num_images: int = 5):
    """提取测试集样本进行 Grad-CAM 可视化并保存。"""
    print("生成 Grad-CAM 可视化...")
    model.eval()
    
    # 抽取各类别单张样本
    sample_images = {}
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            lbl_idx = lbl.item()
            if lbl_idx not in sample_images and len(sample_images) < num_images:
                sample_images[lbl_idx] = img
        if len(sample_images) == num_images:
            break

    # 实例化 Grad-CAM (调用 model.py 中暴露的 target layer)
    target_layer = model.get_cam_layer()
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    
    # ImageNet 逆归一化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, (label, img_tensor) in enumerate(sample_images.items()):
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 将 Tensor 转换为可显示的原始图像
        orig_img = img_tensor.permute(1, 2, 0).numpy()
        orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)

        # 生成 CAM
        cam, pred_class = grad_cam.generate(input_tensor)
        
        # 伪彩色热力图叠加
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        superimposed_img = heatmap * 0.4 + orig_img
        superimposed_img = np.clip(superimposed_img, 0, 1)

        # 绘图
        axes[0, idx].imshow(orig_img)
        axes[0, idx].set_title(f"True Label: {label}")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(superimposed_img)
        axes[1, idx].set_title(f"Pred Label: {pred_class}")
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_samples.svg'), format='svg')
    plt.close()
    
    # 清理 Hook 以防止显存泄漏
    grad_cam.remove_hooks()
    print(f"[*] Grad-CAM 结果已保存至: {output_dir}/gradcam_samples.svg")


def main():
    # ---------------- 基础配置 ----------------
    data_dir = './cancer'
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = 32
    num_epochs = 15
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")

    # ---------------- 数据加载 ----------------
    print("构建数据流水线...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=data_dir, 
        batch_size=batch_size, 
        num_workers=4
    )
    num_classes = len(class_names)
    print(f"检测到 {num_classes} 个类别: {class_names}")

    # ---------------- 模型构建 ----------------
    # 支持加载本地预训练权重进行初始化
    # local_weights = 'pretrained_model/resnet18-f37072fd.pth' 
    model = HistopathologyResNet(num_classes=num_classes, local_weights_path=None).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # ---------------- 训练与评估 ----------------
    best_weights_path = train_and_evaluate(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=num_epochs, 
        device=device,
        output_dir=output_dir
    )

    # ---------------- 测试集验证 ----------------
    print("加载最佳模型进行测试...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    
    test_model(
        model=model, 
        test_loader=test_loader, 
        device=device, 
        num_classes=num_classes, 
        class_names=class_names, 
        output_dir=output_dir
    )

    # ---------------- 解释性分析 ----------------
    run_gradcam(model=model, test_loader=test_loader, device=device, output_dir=output_dir)
    print("全流程执行完毕。")


if __name__ == "__main__":
    # 固定随机种子以保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    main()
