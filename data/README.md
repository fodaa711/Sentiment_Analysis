# Twitter Sentiment Analysis Dataset

This directory is intended for storing the datasets used in the sentiment analysis project.

## Datasets

The project uses the following datasets:

1. **Twitter Entity Sentiment Analysis dataset**
   - Source: Kaggle
   - URL: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
   - Files:
     - twitter_training.csv
     - twitter_validation.csv

2. **Tweet Sentiment Extraction dataset**
   - Source: Kaggle
   - URL: https://www.kaggle.com/competitions/tweet-sentiment-extraction
   - Files:
     - train.csv
     - test.csv
     - sample_submission.csv

## How to Download

### Option 1: Using Kaggle API

1. Install the Kaggle API:
   ```
   pip install kaggle
   ```

2. Configure your Kaggle API credentials:
   - Go to your Kaggle account settings
   - Click on "Create New API Token"
   - Save the kaggle.json file to ~/.kaggle/kaggle.json
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Download the datasets:
   ```
   # Twitter Entity Sentiment Analysis
   kaggle datasets download jp797498e/twitter-entity-sentiment-analysis
   unzip twitter-entity-sentiment-analysis.zip -d data/

   # Tweet Sentiment Extraction
   kaggle competitions download -c tweet-sentiment-extraction
   unzip tweet-sentiment-extraction.zip -d data/
   ```

### Option 2: Using KaggleHub

You can also use the kagglehub package to download the datasets programmatically:

```python
import kagglehub

# Download Twitter Entity Sentiment Analysis dataset
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
print("Path to dataset files:", path)
```

### Option 3: Manual Download

1. Go to the dataset URLs mentioned above
2. Click on the "Download" button
3. Extract the downloaded zip files
4. Place the CSV files in this directory

## Data Format

### Twitter Entity Sentiment Analysis

- **twitter_training.csv** and **twitter_validation.csv**:
  - `text`: The tweet text
  - `sentiment`: The sentiment label (negative, neutral, positive)
  - `entity`: The entity being discussed in the tweet

### Tweet Sentiment Extraction

- **train.csv**:
  - `text`: The tweet text
  - `sentiment`: The sentiment label (negative, neutral, positive)
  - `selected_text`: The part of the text that supports the sentiment

- **test.csv**:
  - `text`: The tweet text
  - `sentiment`: The sentiment label

## Data Preprocessing

The data preprocessing steps are implemented in `utils/data_processing.py`. The main steps include:

1. Text cleaning (removing URLs, user mentions, etc.)
2. Tokenization
3. Optional lemmatization
4. Optional stopword removal

## Note

Due to Kaggle's terms of service, the datasets are not included in this repository. Please download them using one of the methods described above.
