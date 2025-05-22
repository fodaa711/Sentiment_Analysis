# Example Sentiment Analysis Predictions

This document showcases example predictions from our sentiment analysis model on various tweets.

## Example 1: Positive Sentiment

**Tweet:**
```
Just got my new iPhone and I absolutely love it! The camera quality is amazing and Face ID works perfectly. Best purchase this year! #Apple #iPhone
```

**Prediction:**
- **Sentiment:** Positive
- **Confidence:** 0.94
- **Analysis:** The model correctly identified the positive sentiment expressed through words like "love", "amazing", and "best purchase". The exclamation marks and positive descriptors strongly indicate enthusiasm and satisfaction.

## Example 2: Negative Sentiment

**Tweet:**
```
Flight delayed for the third time today. Been stuck at the airport for 5 hours with no updates from the airline. Customer service is non-existent. Never flying with them again.
```

**Prediction:**
- **Sentiment:** Negative
- **Confidence:** 0.89
- **Analysis:** The model correctly identified the negative sentiment through phrases like "delayed", "stuck", "no updates", and "never flying with them again". The frustration and dissatisfaction are clearly expressed.

## Example 3: Neutral Sentiment

**Tweet:**
```
The weather forecast predicts rain tomorrow with temperatures around 65°F. Remember to bring an umbrella if you're heading out.
```

**Prediction:**
- **Sentiment:** Neutral
- **Confidence:** 0.78
- **Analysis:** The model correctly identified the neutral sentiment as the tweet is simply stating factual information about weather without expressing positive or negative emotions.

## Example 4: Mixed Sentiment (Challenging)

**Tweet:**
```
The movie had amazing special effects and cinematography, but the plot was confusing and the characters were poorly developed. Not sure if I'd recommend it.
```

**Prediction:**
- **Sentiment:** Neutral
- **Confidence:** 0.52
- **Analysis:** This was a challenging example with mixed sentiment. The model predicted neutral sentiment with lower confidence, recognizing both positive aspects ("amazing special effects") and negative aspects ("confusing", "poorly developed").

## Example 5: Sarcasm (Challenging)

**Tweet:**
```
Oh great, another Monday morning with traffic jams and a broken coffee machine. Just what I needed to start my week!
```

**Prediction:**
- **Sentiment:** Negative
- **Confidence:** 0.67
- **Analysis:** Despite the sarcastic "great" and "just what I needed", the model correctly identified the underlying negative sentiment. The lower confidence reflects the challenge of detecting sarcasm.

## Performance Analysis

The model performs well on clear positive and negative expressions but shows lower confidence on mixed sentiment and sarcastic content. Future improvements could include:

1. Training with more examples of sarcasm and mixed sentiment
2. Incorporating context awareness
3. Adding emoji understanding for sentiment cues
4. Fine-tuning on domain-specific data

## Visualization

For each prediction, the model generates probability distributions across the three sentiment classes. Below is a sample visualization of prediction probabilities:

```
Example: "Just got my new iPhone and I absolutely love it!"
Probabilities: Negative=0.02, Neutral=0.04, Positive=0.94
```

![Sample Probability Distribution](../output/example_1_probs.png)

*Note: The actual visualization would be generated when running the evaluation script.*
