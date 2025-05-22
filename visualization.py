"""
Visualization utilities for sentiment analysis.
This module contains functions for visualizing data distributions,
model performance, and attention weights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_sentiment_distribution(df, label_col='sentiment', figsize=(10, 6)):
    """
    Plot the distribution of sentiment labels in the dataset.
    
    Args:
        df: Pandas DataFrame with sentiment labels
        label_col: Column name for sentiment labels
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Count sentiment labels
    sentiment_counts = df[label_col].value_counts().sort_index()
    
    # Create bar plot
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 50, str(count), ha='center')
    
    # Set labels and title
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_training_history(history, figsize=(12, 5)):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Dictionary with training history
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size (width, height)
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Set labels and title
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_attention_weights(text, attention_weights, tokenizer, figsize=(12, 8)):
    """
    Visualize attention weights for a text input.
    
    Args:
        text: Input text string
        attention_weights: Attention weights tensor from model
        tokenizer: Tokenizer instance
        figsize: Figure size (width, height)
    """
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    
    # Get attention weights (assuming shape [num_heads, seq_len, seq_len])
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    num_heads = attention_weights.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=num_heads // 2, 
        ncols=2, 
        figsize=figsize, 
        sharex=True, 
        sharey=True
    )
    axes = axes.flatten()
    
    # Plot attention heatmap for each head
    for i in range(num_heads):
        # Get attention weights for this head
        head_weights = attention_weights[i, :len(tokens), :len(tokens)]
        
        # Plot heatmap
        sns.heatmap(
            head_weights, 
            ax=axes[i],
            cmap='viridis',
            xticklabels=tokens,
            yticklabels=tokens
        )
        
        # Set title
        axes[i].set_title(f'Head {i+1}')
        
        # Rotate x-axis labels
        axes[i].set_xticklabels(
            axes[i].get_xticklabels(), 
            rotation=45, 
            ha='right'
        )
    
    # Set overall title
    plt.suptitle('Attention Weights', fontsize=16)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def plot_prediction_probabilities(probabilities, class_names=None, figsize=(8, 5)):
    """
    Visualize prediction probabilities for a single text input.
    
    Args:
        probabilities: Array of class probabilities
        class_names: List of class names
        figsize: Figure size (width, height)
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = plt.bar(class_names, probabilities, color=colors)
    
    # Add probability values on top of bars
    for bar, prob in zip(bars, probabilities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{prob:.2f}',
            ha='center',
            fontsize=12
        )
    
    # Set labels and title
    plt.title('Prediction Probabilities', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
