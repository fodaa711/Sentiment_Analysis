"""
Training utilities for sentiment analysis models.
This module contains functions for training and evaluating
sentiment analysis models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import time


class SentimentDataset(Dataset):
    """
    Dataset class for sentiment analysis.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens), torch.tensor(label, dtype=torch.long)


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=32, 
                       max_length=512, num_workers=4, pin_memory=True):
    """
    Create DataLoader objects for training, validation, and testing.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SentimentDataset(
        train_df['processed_text'].tolist(),
        train_df['encoded_label'].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = SentimentDataset(
        val_df['processed_text'].tolist(),
        val_df['encoded_label'].tolist(),
        tokenizer,
        max_length
    )
    
    test_dataset = SentimentDataset(
        test_df['processed_text'].tolist(),
        test_df['encoded_label'].tolist(),
        tokenizer,
        max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch
    )
    
    return train_loader, val_loader, test_loader


def collate_batch(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (tokens, label) tuples
        
    Returns:
        Padded tokens tensor and labels tensor
    """
    tokens, labels = zip(*batch)
    
    # Pad sequences
    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return tokens_padded, labels


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch.
    
    Args:
        model: Model instance
        dataloader: DataLoader for training data
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to use (cuda or cpu)
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for tokens, labels in progress_bar:
        tokens, labels = tokens.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation or test data.
    
    Args:
        model: Model instance
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use (cuda or cpu)
        
    Returns:
        Average loss, accuracy, and predictions
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Evaluating"):
            tokens, labels = tokens.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, 
                weight_decay=1e-5, patience=3, save_dir=None):
    """
    Train model with early stopping.
    
    Args:
        model: Model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        save_dir: Directory to save model checkpoints
        
    Returns:
        Trained model and training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create save directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                
                print(f"Model saved to {os.path.join(save_dir, 'best_model.pt')}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model if saved
    if save_dir and os.path.exists(os.path.join(save_dir, 'best_model.pt')):
        checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return model, history


def predict_sentiment(text, model, tokenizer, device=None):
    """
    Predict sentiment for a single text input.
    
    Args:
        text: Input text string
        model: Trained model instance
        tokenizer: Tokenizer instance
        device: Device to use (cuda or cpu)
        
    Returns:
        Predicted sentiment class and probabilities
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
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
    
    return predicted_sentiment, probs
