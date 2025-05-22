"""
Transformer model implementation for sentiment analysis.
This module contains the implementation of the transformer-based model
for sentiment classification of tweets.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SentimentTransformer(nn.Module):
    """
    Transformer model for sentiment analysis.
    """
    def __init__(self, vocab_size, dim_model=256, dim_feedforward=1024, 
                 num_layers=6, num_heads=8, dropout=0.1, max_len=512, 
                 num_classes=3, batch_first=True):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_first = batch_first
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, num_classes)
        )
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for the model."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] if batch_first=True else [seq_len, batch_size]
            src_mask: Optional tensor, shape [seq_len, seq_len]
            src_key_padding_mask: Optional tensor, shape [batch_size, seq_len]
        """
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        
        if not self.batch_first:
            src = src.transpose(0, 1)  # [batch_size, seq_len, dim_model]
            
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        
        # Global average pooling
        output = output.mean(dim=1)  # [batch_size, dim_model]
        
        # Classification
        output = self.classifier(output)  # [batch_size, num_classes]
        
        return output
    
    def predict(self, text_tokens):
        """
        Make a prediction for a single tokenized text.
        
        Args:
            text_tokens: Tensor, shape [1, seq_len]
        
        Returns:
            Predicted sentiment class (0: negative, 1: neutral, 2: positive)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(text_tokens)
            predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class
