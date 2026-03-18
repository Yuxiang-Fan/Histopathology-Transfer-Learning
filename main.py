import torch
from dataset import get_dataloaders
from model import build_resnet

# ==========================================
# 1. 配置与参数
# ==========================================
DATA_DIR = 'data/cancer'
NUM_CLASSES = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    # 1. 准备数据
    print("Loading datasets...")
    # 注意：运行前需确保 data/cancer 目录下有 5 个子类文件夹
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Classes found: {class_names}")

    # 2. 构建模型 (自动拉取官方预训练权重)
    print("Initializing model...")
    model = build_resnet(NUM_CLASSES)
    model = model.to(DEVICE)

    # 3. 提示信息
    print("Architecture loaded successfully. Ready for training.")
    print(
        "Note: Run the associated Jupyter Notebook for the full training loop, evaluation, and Grad-CAM visualization.")


if __name__ == '__main__':
    main()