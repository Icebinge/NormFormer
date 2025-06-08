import torch
import torch.nn as nn
from models.normformer import Transformer
from models.normActive import RowNormActivation

class SimpleTransformerNet(nn.Module):
    def __init__(self, config):
        super(SimpleTransformerNet, self).__init__()
        self.config = config
        
        # 使用完整的Transformer (包含patch embedding和位置编码)
        self.transformer = Transformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            patch_size=config.patch_size
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            # RowNormActivation(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.transformer.patch_embed.weight)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        # 输入 x: (batch_size, 1, 28, 28)
        
        # 通过Transformer (包含patch embedding、位置编码和编码器层)
        x = self.transformer(x)
        
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