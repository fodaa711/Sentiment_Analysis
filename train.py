#!/usr/bin/env python
"""
Training script for sentiment analysis model.
This script trains a transformer-based model for sentiment analysis
on Twitter data.
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import local modules
from models.transformer import SentimentTransformer
from models.tokenizer import SentimentTokenizer
from utils.data_processing import prepare_dataset, split_dataset
from utils.training import create_dataloaders, train_model
from utils.visualization import plot_training_history, plot_sentiment_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    parser.add_argument('--text_col', type=str, default='text',
                        help='Column name for text data')
    parser.add_argument('--label_col', type=str, default='sentiment',
                        help='Column name for sentiment labels')
    
    # Tokenizer arguments
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to pre-trained tokenizer model')
    parser.add_argument('--vocab_size', type=int, default=12000,
                        help='Vocabulary size for tokenizer')
    
    # Model arguments
    parser.add_argument('--dim_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Feedforward dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoint')
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot training results')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Prepare dataset
    print("Preparing dataset...")
    df_processed = prepare_dataset(df, args.text_col, args.label_col)
    
    # Split dataset
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df_processed)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Plot sentiment distribution
    if args.plot_results:
        print("Plotting sentiment distribution...")
        plot_sentiment_distribution(df_processed, 'encoded_label')
        plt.savefig(os.path.join(args.output_dir, 'sentiment_distribution.png'))
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = SentimentTokenizer(args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path}")
    else:
        tokenizer = SentimentTokenizer(vocab_size=args.vocab_size)
        print(f"Created new tokenizer with vocab size {args.vocab_size}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_len,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = SentimentTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        dim_model=args.dim_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_len=args.max_len
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Training model...")
    save_dir = args.output_dir if args.save_model else None
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=save_dir
    )
    
    # Plot training history
    if args.plot_results:
        print("Plotting training history...")
        plot_training_history(history)
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    from utils.training import evaluate
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Plot confusion matrix
    if args.plot_results:
        print("Plotting confusion matrix...")
        from utils.visualization import plot_confusion_matrix
        plot_confusion_matrix(test_labels, test_preds)
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print("Training completed!")


if __name__ == "__main__":
    main()
