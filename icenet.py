import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import time
import os
from torch.cuda.amp import GradScaler, autocast
from normformer import EncoderLayer

class Config:
    """集中管理所有超参数"""
    seed = 42
    num_classes = 10
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    patch_size = 7
    
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    max_grad_norm = 1.0
    
    train_ratio = 0.9
    num_workers = 4
    use_amp = True  # 自动混合精度训练
    
    save_dir = "icenet_checkpoints"
    model_name = "transformer_mnist"

class SimpleTransformerNet(nn.Module):
    def __init__(self, config):
        super(SimpleTransformerNet, self).__init__()
        self.config = config
        
        # 图像预处理层
        self.patch_embed = nn.Conv2d(1, config.d_model, 
                                    kernel_size=config.patch_size, 
                                    stride=config.patch_size)
        
        # 位置编码
        num_patches = (28 // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.d_model))
        
        # Transformer 编码器 (使用之前修改后的纯编码器版本)
        self.transformer = nn.Sequential(*[
            EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # 输入 x: (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # 图像分块
        x = self.patch_embed(x)  # (batch_size, d_model, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, d_model)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过Transformer编码器
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        return self.classifier(x)

class Trainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        self._set_seed()
        
        # 准备数据
        self._prepare_data()
        
        # 初始化模型
        self.model = SimpleTransformerNet(config).to(self.device)
        
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
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
    def _set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _prepare_data(self):
        """准备数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 加载数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # 划分训练集和验证集
        train_size = int(self.config.train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
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
            with autocast(enabled=self.config.use_amp):
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
            checkpoint = torch.load(filename)
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
    config = Config()
    trainer = Trainer(config)
    history = trainer.train()

if __name__ == '__main__':
    main()