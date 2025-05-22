#!/usr/bin/env python
"""
Evaluation script for sentiment analysis model.
This script evaluates a trained transformer-based model for sentiment analysis
on Twitter data and generates predictions.
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
from utils.training import create_dataloaders, evaluate
from utils.visualization import plot_confusion_matrix, plot_prediction_probabilities


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset CSV file')
    parser.add_argument('--text_col', type=str, default='text',
                        help='Column name for text data')
    parser.add_argument('--label_col', type=str, default='sentiment',
                        help='Column name for sentiment labels')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to tokenizer model')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to CSV')
    parser.add_argument('--plot_results', action='store_true',
                        help='Plot evaluation results')
    parser.add_argument('--examples', type=int, default=5,
                        help='Number of example predictions to show')
    
    return parser.parse_args()


def predict_examples(model, tokenizer, texts, device):
    """
    Generate predictions for example texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        texts: List of text strings
        device: Device to use
        
    Returns:
        List of (text, sentiment, probabilities) tuples
    """
    model.eval()
    results = []
    
    for text in texts:
        # Tokenize text
        tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([tokens]).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(tokens_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Map class index to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        predicted_sentiment = sentiment_map[predicted_class]
        
        # Get probabilities for each class
        probs = probabilities[0].cpu().numpy()
        
        results.append((text, predicted_sentiment, probs))
    
    return results


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SentimentTokenizer(args.tokenizer_path)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model
    model = SentimentTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        dim_model=256,  # These should match the trained model
        dim_feedforward=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_len=args.max_len
    )
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
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
    print(f"Test set shape: {test_df.shape}")
    
    # Create test dataloader
    print("Creating test dataloader...")
    test_dataset = utils.training.SentimentDataset(
        test_df['processed_text'].tolist(),
        test_df['encoded_label'].tolist(),
        tokenizer,
        args.max_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.training.collate_batch
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Plot confusion matrix
    if args.plot_results:
        print("Plotting confusion matrix...")
        plot_confusion_matrix(test_labels, test_preds)
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save predictions
    if args.save_predictions:
        print("Saving predictions...")
        test_df['predicted_sentiment'] = test_preds
        test_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    # Generate example predictions
    print("Generating example predictions...")
    example_texts = test_df['processed_text'].sample(args.examples).tolist()
    example_results = predict_examples(model, tokenizer, example_texts, device)
    
    # Print example predictions
    print("\nExample Predictions:")
    for i, (text, sentiment, probs) in enumerate(example_results):
        print(f"\nExample {i+1}:")
        print(f"Text: {text}")
        print(f"Predicted sentiment: {sentiment}")
        print(f"Probabilities: Negative={probs[0]:.4f}, Neutral={probs[1]:.4f}, Positive={probs[2]:.4f}")
        
        # Plot prediction probabilities
        if args.plot_results:
            plot_prediction_probabilities(probs)
            plt.savefig(os.path.join(args.output_dir, f'example_{i+1}_probs.png'))
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
