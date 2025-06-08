import torch
import torch.nn as nn
from models.normformer import PositionalEncoding

class OriginalTransformerBlock(nn.Module):
    def __init__(self, config):
        super(OriginalTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, 
            config.num_heads, 
            dropout=config.dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),  # 使用GELU代替ReLU
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class OriginalTransformerNet(nn.Module):
    def __init__(self, config):
        super(OriginalTransformerNet, self).__init__()
        self.config = config
        
        # 图像预处理层
        self.patch_embed = nn.Conv2d(
            1, config.d_model, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        
        # 位置编码
        num_patches = (28 // config.patch_size) ** 2
        self.positional_encoding = PositionalEncoding(config.d_model, max_len=num_patches)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            OriginalTransformerBlock(config) for _ in range(config.num_layers)
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
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # 输入 x: (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # 图像分块
        x = self.patch_embed(x)  # (batch_size, d_model, 4, 4)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 通过Transformer编码器层
        for block in self.transformer_blocks:
            x = block(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        return self.classifier(x)

    def print_model_summary(self):
        """打印模型结构"""
        try:
            from torchinfo import summary
            print("\nDetailed Model Summary:")
            summary(self, 
                    input_size=(self.config.batch_size, 1, 28, 28),
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                    verbose=1)
        except ImportError:
            print("torchinfo 未安装，请使用: pip install torchinfo") 