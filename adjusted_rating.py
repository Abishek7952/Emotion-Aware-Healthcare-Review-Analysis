import pandas as pd

# Load data
sentiment_df = pd.read_csv("medical_sentiment_mismatch.csv")
emotion_df = pd.read_csv("emotion_analysis_output.csv")

# Merge on 'text'
merged_df = pd.merge(sentiment_df, emotion_df[['text', 'top_emotions']], on='text', how='left')

# Define emotion weights
emotion_weights = {
    'anger': -1.0, 'sadness': -0.8, 'disgust': -0.9, 'fear': -0.6,
    'disappointment': -0.5, 'remorse': -0.7, 'grief': -1.0,
    'joy': 0.6, 'gratitude': 0.5, 'love': 0.7,
    'relief': 0.4, 'optimism': 0.4, 'excitement': 0.3, 'pride': 0.3
}

# Function to compute new rating only for mismatches
def compute_adjusted_rating(row):
    if not row['sentiment_mismatch']:
        return row['stars']  # No change if no mismatch

    original = row['stars']
    emotions = [e.strip().lower() for e in str(row['top_emotions']).split(',')]
    total_adjustment = sum(emotion_weights.get(em, 0) for em in emotions)

    # Final rating clamped between 1 and 5
    return min(5, max(1, round(original + total_adjustment, 1)))

# Apply logic
merged_df['adjusted_rating'] = merged_df.apply(compute_adjusted_rating, axis=1)

# Save final file
merged_df.to_csv("final_adjusted_ratings.csv", index=False)
print("✅ Final output with emotion-weighted adjustments saved as 'final_adjusted_ratings.csv'")
