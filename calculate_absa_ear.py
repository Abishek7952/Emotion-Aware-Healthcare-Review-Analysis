import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_ASPECT_FILE = "aspect_level_emotion_analysis.csv"
INPUT_MISMATCH_FILE = "medical_sentiment_mismatch.csv" 
OUTPUT_FILE = "final_absa_adjusted_ratings.csv" # This file will now have the new column

# Same weights as before
EMOTION_WEIGHTS = {
    'anger': -1.0, 'disgust': -0.9, 'fear': -0.6, 'sadness': -0.8, 'disappointment': -0.7,
    'annoyance': -0.5, 'remorse': -0.7, 'grief': -1.0, 'embarrassment': -0.2, 'nervousness': -0.3,
    'joy': 0.8, 'gratitude': 0.7, 'love': 0.9, 'admiration': 0.6, 'approval': 0.4, 'caring': 0.5,
    'excitement': 0.3, 'optimism': 0.4, 'relief': 0.5, 'pride': 0.3
}
# ------------------------

print(f"Loading aspect data from {INPUT_ASPECT_FILE}...")
aspect_df = pd.read_csv(INPUT_ASPECT_FILE)

print(f"Loading original mismatch data from {INPUT_MISMATCH_FILE}...")
df = pd.read_csv(INPUT_MISMATCH_FILE)

# --- 1. Calculate Total Adjustment Score (Same as before) ---
review_adjustments = {}
print("Calculating aspect-based emotional adjustments...")
for _, row in tqdm(aspect_df.iterrows(), total=aspect_df.shape[0]):
    review_id = row['review_id']
    emotions = str(row['sentence_top_emotions']).split(', ')
    aspect_score = sum(EMOTION_WEIGHTS.get(em.strip().lower(), 0) for em in emotions)
    
    review_adjustments[review_id] = review_adjustments.get(review_id, 0.0) + aspect_score

df['aspect_emotion_adjustment'] = df.index.map(review_adjustments).fillna(0.0)

# --- 2. Calculate Original "Anchored" Rating (Same as before) ---
def compute_anchored_ear(row):
    if row['sentiment_mismatch'] and row['aspect_emotion_adjustment'] != 0:
        original = row['stars']
        adjustment = row['aspect_emotion_adjustment']
        return round(min(5, max(1, original + adjustment)), 1)
    else:
        return row['stars']

df['absa_adjusted_rating'] = df.apply(compute_anchored_ear, axis=1)

# --- 3. (❗❗ NEW ❗❗) Calculate "De-Anchored" Emotion-Only Rating ---
def compute_emotion_only_rating(adjustment_score):
    """
    Creates a new rating from a neutral 3.0 base,
    using only the emotion adjustment.
    """
    # If no aspects were found, we can't give an emotion score.
    # We will use the original star rating as a fallback.
    if adjustment_score == 0.0:
        return None # We will fill this in later
    
    base_rating = 3.0 # Start from a neutral base
    new_rating = base_rating + adjustment_score
    
    return round(min(5, max(1, new_rating)), 1)

print("Applying final adjustments to create *Emotion-Only* Rating...")
df['emotion_only_rating'] = df['aspect_emotion_adjustment'].apply(compute_emotion_only_rating)

# For any review where we couldn't calculate an emotion_only_rating (no aspects),
# we fall back to using its original star rating.
df['emotion_only_rating'] = df['emotion_only_rating'].fillna(df['stars'])

# --- 4. Save ---
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Success! File saved to: {OUTPUT_FILE}")
print("\nPreview of new 'emotion_only_rating':")
print(df[['text', 'stars', 'aspect_emotion_adjustment', 'emotion_only_rating']].head())