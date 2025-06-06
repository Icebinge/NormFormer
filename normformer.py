import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from normActive import RowNormActivation

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

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
        
        # Reshape for normalization
        batch_size, num_heads, seq_len, d_k = Q.shape
        Q_reshaped = Q.reshape(-1, d_k)  # (batch_size * num_heads * seq_len, d_k)
        K_reshaped = K.reshape(-1, d_k)  # (batch_size * num_heads * seq_len, d_k)
        
        # Normalize using RowNormActivation
        Q_norm = self.norm_activation(Q_reshaped).view(batch_size, num_heads, seq_len, d_k)
        K_norm = self.norm_activation(K_reshaped).view(batch_size, num_heads, seq_len, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)
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
        x = self.norm_activation(x)
        x = self.dropout(x)
        return self.linear2(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_activation = RowNormActivation()
        
    def forward(self, x):
        # Self attention
        attn_output = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm_activation(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm_activation(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.norm_activation = RowNormActivation()
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # Encoder only
        src_embedded = self.dropout(self.positional_encoding(self.src_embedding(src.long()) * math.sqrt(self.d_model)))
        src_embedded = self.norm_activation(src_embedded)
        enc_output = src_embedded
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