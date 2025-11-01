import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_ASPECT_FILE = "aspect_level_emotion_analysis.csv"
INPUT_MISMATCH_FILE = "medical_sentiment_mismatch.csv" # To get all reviews
OUTPUT_FILE = "final_absa_adjusted_ratings.csv"

# ❗ These weights are still heuristics, but they are now applied
# to specific aspects, not the whole review. This is the key change.
# You can (and should) tune these for your paper.
EMOTION_WEIGHTS = {
    # Negative
    'anger': -1.0, 
    'disgust': -0.9,
    'fear': -0.6,
    'sadness': -0.8, 
    'disappointment': -0.7,
    'annoyance': -0.5,
    'remorse': -0.7,
    'grief': -1.0,
    
    # Positive
    'joy': 0.8, 
    'gratitude': 0.7, 
    'love': 0.9,
    'admiration': 0.6,
    'approval': 0.4,
    'caring': 0.5,
    'excitement': 0.3,
    'optimism': 0.4,
    'relief': 0.5,
    'pride': 0.3,
    
    # Neutral/Other (we can ignore these by setting to 0)
    'confusion': -0.1,
    'curiosity': 0.0,
    'desire': 0.0,
    'embarrassment': -0.2,
    'nervousness': -0.3,
    'realization': 0.0,
    'surprise': 0.0,
    'neutral': 0.0
}
# ------------------------

print(f"Loading aspect data from {INPUT_ASPECT_FILE}...")
try:
    aspect_df = pd.read_csv(INPUT_ASPECT_FILE)
except FileNotFoundError:
    print(f"Error: {INPUT_ASPECT_FILE} not found. Did you run absa_emotion_analysis.py first?")
    exit()

print(f"Loading original mismatch data from {INPUT_MISMATCH_FILE}...")
try:
    # We use the original mismatch file as the "base"
    df = pd.read_csv(INPUT_MISMATCH_FILE)
except FileNotFoundError:
    print(f"Error: {INPUT_MISMATCH_FILE} not found. Did you run sentiment_mismatch_analysis.py first?")
    exit()

# This dict will store the total adjustment for each review_id
review_adjustments = {}

print("Calculating aspect-based emotional adjustments...")
# Iterate through the aspect file, which has multiple rows per review
for _, row in tqdm(aspect_df.iterrows(), total=aspect_df.shape[0]):
    review_id = row['review_id']
    emotions = str(row['sentence_top_emotions']).split(', ')
    
    # Calculate the total emotional score for THIS ONE ASPECT
    aspect_score = 0
    for em in emotions:
        aspect_score += EMOTION_WEIGHTS.get(em.strip().lower(), 0)
    
    # Add this aspect's score to the review's total adjustment
    if review_id not in review_adjustments:
        review_adjustments[review_id] = 0.0
    review_adjustments[review_id] += aspect_score

# Now, map these total adjustments back to the original dataframe
df['aspect_emotion_adjustment'] = df.index.map(review_adjustments).fillna(0.0)

# Apply the adjustment
def compute_absa_ear(row):
    # Only apply adjustment if the review was a mismatch AND we found aspects in it
    if row['sentiment_mismatch'] and row['aspect_emotion_adjustment'] != 0:
        original = row['stars']
        adjustment = row['aspect_emotion_adjustment']
        
        # Clamp the final rating between 1 and 5
        return round(min(5, max(1, original + adjustment)), 1)
    else:
        # If no mismatch or no aspects found, the rating stays the same
        return row['stars']

print("Applying final adjustments to create ABSA-EAR...")
df['absa_adjusted_rating'] = df.apply(compute_absa_ear, axis=1)

# Save the final, comprehensive file
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Success! Your new 'ABSA-Adjusted Rating (EAR)' is complete.")
print(f"Final data saved to: {OUTPUT_FILE}")

# Show a preview of the changes
print("\nPreview of adjustments:")
changed_df = df[df['stars'] != df['absa_adjusted_rating']]
print(changed_df[['text', 'stars', 'aspect_emotion_adjustment', 'absa_adjusted_rating']].head())