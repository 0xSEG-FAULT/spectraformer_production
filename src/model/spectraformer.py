"""
SpectraFormer: Transformer-based model for hyperspectral crop classification
Based on the paper architecture with positional encoding and multi-head attention
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src):
    # Expect src as (seq_len, batch, d_model) (seq-first)
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class SpectraFormer(nn.Module):
    """SpectraFormer model for hyperspectral crop classification"""
    
    def __init__(self, config):
        super(SpectraFormer, self).__init__()
        self.config = config
        
        # Input projection: from 331 features to hidden_dim
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, max_len=500)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.mlp_dim,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)
        
    def forward(self, x):
        """
        x: (batch_size, 331) - spectral features
        return: (batch_size, num_classes) - logits
        """
        # Project input
        x = self.input_projection(x)  # (batch, hidden_dim)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)  # (batch, 1, hidden_dim)
        
        # Transpose for transformer (seq_len, batch, hidden_dim)
        x = x.transpose(0, 1)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Transpose back to batch-first
        x = x.transpose(0, 1)  # (batch, 1, hidden_dim)
        
        # Average pooling over sequence (take first token)
        x = x[:, 0, :]  # (batch, hidden_dim)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)  # (batch, num_classes)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)