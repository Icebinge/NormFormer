import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.normActive import RowNormActivation

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 使用可学习的位置编码参数
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)  # 使用与原始Transformer相同的初始化方式

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_activation = RowNormActivation()
        
    def scaled_dot_product_attention(self, Q, K, V):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        
        # Normalize using RowNormActivation
        Q_norm = self.norm_activation(Q)
        K_norm = self.norm_activation(K)
        
        # Compute attention scores
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        attention_weights = scores ** 2  # Replace softmax with square operation
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output = self.scaled_dot_product_attention(Q, K, V)
        
        # Reshape and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_activation = RowNormActivation()
        
    def forward(self, x):
        x = self.linear1(x)
        # x = self.norm_activation(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.linear2(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_activation = RowNormActivation()
        self.norm1 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self attention
        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        # x = self.norm_activation(x)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        # x = self.norm_activation(x)
        x = self.norm1(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8,
                 num_encoder_layers=6, d_ff=2048, dropout=0.1,
                 patch_size=7):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 图像预处理层
        self.patch_embed = nn.Conv2d(
            1, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 位置编码
        num_patches = (28 // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_patches)
        self.norm_activation = RowNormActivation()
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        # self._init_weights()
        
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
        
        # 通过编码器层
        enc_output = x
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
            
        return enc_output

# Example usage
if __name__ == "__main__":
    # Create a small transformer for demonstration
    src_vocab_size = 1000
    d_model = 512
    num_heads = 8
    
    transformer = Transformer(src_vocab_size, d_model, num_heads)
    
    # Create dummy input
    batch_size = 2
    src_seq_length = 10
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_length))
    
    # Forward pass
    output = transformer(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")