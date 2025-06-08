import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from tqdm import tqdm
import os
from torch.amp import GradScaler, autocast
import math
import yaml
from dataclasses import dataclass
from typing import Dict, Any
from datasets.mnist_dataset import MNISTDataset
from models.norm_transformer import SimpleTransformerNet

@dataclass
class Config:
    """配置参数类"""
    model: Dict[str, Any]
    training: Dict[str, Any]
    system: Dict[str, Any]
    save: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def __getattr__(self, name: str) -> Any:
        """允许直接访问配置项"""
        for section in [self.model, self.training, self.system, self.save]:
            if name in section:
                return section[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class Trainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        self._set_seed()
        
        # 准备数据
        data_loader = MNISTDataset(config)
        loaders = data_loader.get_data_loaders()
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
        
        # 初始化模型
        self.model = SimpleTransformerNet(config).to(self.device)
        
        # 打印模型结构
        self.print_model_summary()
        
        # 优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        print(f"当前设备: {self.device}")
        
    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def print_model_summary(self):
        """打印模型结构"""
        self.model.print_model_summary()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 混合精度训练
            with autocast(device_type=self.device.type, enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 计算指标
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        return 100. * val_correct / val_total
    
    def test(self):
        """测试模型"""
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        return 100. * test_correct / test_total
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_acc': self.best_val_acc if hasattr(self, 'best_val_acc') else 0.0
        }
        
        filename = os.path.join(self.config.save_dir, f'{self.config.model_name}_last.pth')
        torch.save(state, filename)
        
        if is_best:
            best_filename = os.path.join(self.config.save_dir, f'{self.config.model_name}_best.pth')
            torch.save(state, best_filename)
    
    def load_checkpoint(self):
        """加载模型检查点"""
        filename = os.path.join(self.config.save_dir, f'{self.config.model_name}_last.pth')
        if os.path.exists(filename):
            checkpoint = torch.load(filename, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            return checkpoint['epoch'], checkpoint.get('best_acc', 0.0)
        return 0, 0.0
    
    def train(self):
        """训练主循环"""
        start_epoch, self.best_val_acc = self.load_checkpoint()
        
        train_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(start_epoch, self.config.num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 验证
            val_acc = self.validate()
            val_accs.append(val_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # 保存最新模型
            self.save_checkpoint(epoch)
            
            # 打印epoch信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.config.num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s')
            
            # 保存训练日志
            with open(os.path.join(self.config.save_dir, 'training_log.txt'), 'a') as log_file:
                log_file.write(f'Epoch {epoch+1}/{self.config.num_epochs} - '
                               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                               f'Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s\n')
        
        # 最终测试
        test_acc = self.test()
        print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc,
            'best_val_acc': self.best_val_acc
        }
        torch.save(history, os.path.join(self.config.save_dir, 'training_history.pth'))
        
        return history

def main():
    config = Config.from_yaml('configs/norm_transformer_config.yaml')
    trainer = Trainer(config)
    history = trainer.train()
    # filename = os.path.join('checkpoints', f'{trainer.config.model_name}_best.pth')
    # if os.path.exists(filename):
    #     checkpoint = torch.load(filename, weights_only=True)
    #     print(checkpoint['state_dict'].keys())
    #     trainer.model.load_state_dict(checkpoint['state_dict'])
    #     trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    #     trainer.scheduler.load_state_dict(checkpoint['scheduler'])
    #     trainer.scaler.load_state_dict(checkpoint['scaler'])
    # trainer.model.eval()
    # test_correct = 0
    # test_total = 0
    
    # with torch.no_grad():
    #     for images, labels in tqdm(trainer.test_loader, desc='Testing'):
    #         images, labels = images.to(trainer.device), labels.to(trainer.device)
    #         outputs = trainer.model(images)
    #         _, predicted = outputs.max(1)
    #         test_total += labels.size(0)
    #         test_correct += predicted.eq(labels).sum().item()
        
    # print(100. * test_correct / test_total)

if __name__ == '__main__':
    main()