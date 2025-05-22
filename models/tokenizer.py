"""
SentencePiece tokenizer wrapper for sentiment analysis.
This module provides a wrapper around the SentencePiece tokenizer
for text preprocessing in the sentiment analysis model.
"""

import sentencepiece as spm
import os
import torch


class SentimentTokenizer:
    """
    Wrapper for SentencePiece tokenizer with additional functionality
    for sentiment analysis preprocessing.
    """
    def __init__(self, model_path=None, vocab_size=12000, model_type="bpe"):
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, input_file, model_prefix, normalization_rule_name="identity",
              character_coverage=0.9995, user_defined_symbols="<cls>"):
        """
        Train a new SentencePiece tokenizer model.
        
        Args:
            input_file: Path to text file for training
            model_prefix: Prefix for the model files
            normalization_rule_name: Normalization rule
            character_coverage: Character coverage percentage
            user_defined_symbols: Special tokens to include
        """
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            normalization_rule_name=normalization_rule_name,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        self.model_path = f"{model_prefix}.model"
        self.load(self.model_path)
        
    def load(self, model_path):
        """
        Load a trained SentencePiece model.
        
        Args:
            model_path: Path to the .model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        
    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        if add_special_tokens:
            return self.sp.encode(text, add_bos=True, add_eos=True)
        else:
            return self.sp.encode(text)
    
    def decode(self, ids):
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        return self.sp.decode(ids)
    
    def tokenize(self, text):
        """
        Tokenize text to subword tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        return self.sp.encode_as_pieces(text)
    
    def batch_encode(self, texts, max_length=None, padding=True, truncation=True):
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Tensor of token IDs with shape [batch_size, seq_len]
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        batch_ids = []
        for text in texts:
            ids = self.encode(text)
            
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
                
            batch_ids.append(torch.tensor(ids))
        
        if padding:
            return torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=0)
        else:
            return batch_ids
    
    def get_vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.
        
        Returns:
            Vocabulary size
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        return self.sp.get_piece_size()
    
    def id_to_token(self, id):
        """
        Convert a token ID to its string representation.
        
        Args:
            id: Token ID
            
        Returns:
            Token string
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        return self.sp.id_to_piece(id)
    
    def token_to_id(self, token):
        """
        Convert a token string to its ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID
        """
        if not self.sp:
            raise ValueError("Tokenizer model not loaded. Call load() first.")
        
        return self.sp.piece_to_id(token)
