"""
Data processing utilities for sentiment analysis.
This module contains functions for cleaning, preprocessing, and preparing
Twitter data for sentiment analysis.
"""

import re
import string
import emoji
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import numpy as np


def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Replace user mentions with token
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Replace hashtags with token + hashtag content
    text = re.sub(r'#(\w+)', r'[HASHTAG] \1', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', lambda m: ' ' + m.group(0).replace(':', '').replace('_', ' ') + ' ', text)
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_wordnet_pos(tag):
    """
    Map POS tag to WordNet POS tag.
    
    Args:
        tag: POS tag from NLTK
        
    Returns:
        WordNet POS tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_text(text):
    """
    Lemmatize text using WordNet lemmatizer.
    
    Args:
        text: Input text string
        
    Returns:
        Lemmatized text string
    """
    lemmatizer = nltk.WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    
    # Get POS tags
    tagged_tokens = pos_tag(word_tokens)
    
    # Lemmatize with appropriate POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]
    
    return ' '.join(lemmatized_tokens)


def remove_stopwords(text, custom_stopwords=None):
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
        custom_stopwords: Optional list of additional stopwords
        
    Returns:
        Text with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    
    return ' '.join(filtered_text)


def preprocess_text(text, remove_stops=False, lemmatize=True, custom_stopwords=None):
    """
    Full preprocessing pipeline for text data.
    
    Args:
        text: Input text string
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize text
        custom_stopwords: Optional list of additional stopwords
        
    Returns:
        Preprocessed text string
    """
    # Clean text
    text = clean_text(text)
    
    # Lemmatize if requested
    if lemmatize:
        text = lemmatize_text(text)
    
    # Remove stopwords if requested
    if remove_stops:
        text = remove_stopwords(text, custom_stopwords)
    
    return text


def prepare_dataset(df, text_col='text', label_col='sentiment', 
                    remove_stops=False, lemmatize=True):
    """
    Prepare dataset for sentiment analysis.
    
    Args:
        df: Pandas DataFrame with text and labels
        text_col: Column name for text data
        label_col: Column name for sentiment labels
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize text
        
    Returns:
        DataFrame with preprocessed text and encoded labels
    """
    # Copy dataframe to avoid modifying original
    df_processed = df.copy()
    
    # Preprocess text
    df_processed['processed_text'] = df_processed[text_col].apply(
        lambda x: preprocess_text(x, remove_stops, lemmatize)
    )
    
    # Encode labels if they're strings
    if df_processed[label_col].dtype == 'object':
        label_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        df_processed['encoded_label'] = df_processed[label_col].map(label_map)
    else:
        df_processed['encoded_label'] = df_processed[label_col]
    
    return df_processed


def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: Pandas DataFrame
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['encoded_label']
    )
    
    # Second split: train and val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state, 
        stratify=train_val_df['encoded_label']
    )
    
    return train_df, val_df, test_df
