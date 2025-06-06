import os
import torch
from torchvision import datasets, transforms

def download_mnist():
    # 创建数据目录
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 下载 MNIST 数据集
    print("开始下载 MNIST 数据集...")
    try:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print("下载完成！")
    except Exception as e:
        print(f"下载过程中出现错误: {str(e)}")
        return False
    
    return True

if __name__ == '__main__':
    download_mnist() 