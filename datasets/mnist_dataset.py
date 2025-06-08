import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any

class MNISTDataset:
    """MNIST数据集加载器"""
    def __init__(self, config: Any):
        self.config = config
        self.transform = self._get_transform()
        
    def _get_transform(self) -> transforms.Compose:
        """获取数据转换"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def _load_datasets(self) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """加载训练集和测试集"""
        train_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.transform
        )
        return train_dataset, test_dataset
    
    def _split_train_val(self, train_dataset: datasets.MNIST) -> Tuple[datasets.MNIST, datasets.MNIST]:
        """划分训练集和验证集"""
        train_size = int(self.config.train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        return random_split(train_dataset, [train_size, val_size])
    
    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """获取数据加载器"""
        # 加载数据集
        train_dataset, test_dataset = self._load_datasets()
        
        # 划分训练集和验证集
        train_dataset, val_dataset = self._split_train_val(train_dataset)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        } 