import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import Counter
from tqdm import tqdm

print("--- Starting Conflict Heatmap Analysis ---")

# --- 1. Define Emotion Categories ---
POSITIVE_EMOTIONS = {'joy', 'gratitude', 'love', 'admiration', 'approval', 'caring', 'excitement', 'optimism', 'relief', 'pride'}
NEGATIVE_EMOTIONS = {'anger', 'disgust', 'fear', 'sadness', 'disappointment', 'annoyance', 'remorse', 'grief', 'embarrassment', 'nervousness'}

# --- 2. Load Data ---
print("Loading data...")
try:
    aspect_df = pd.read_csv("aspect_level_emotion_analysis.csv")
    review_df = pd.read_csv("medical_sentiment_mismatch.csv")
    
    if 'review_id' in review_df.columns:
        review_df = review_df.drop(columns=['review_id'])
    review_df = review_df.reset_index().rename(columns={'index': 'review_id'})
    
except FileNotFoundError:
    print("Error: Could not find 'aspect_level_emotion_analysis.csv' or 'medical_sentiment_mismatch.csv'.")
    exit()

# --- 3. Merge Data ---
print("Merging dataframes...")
merged_df = pd.merge(
    aspect_df,
    review_df, 
    on='review_id'
)

# --- 4. Filter for "Hidden Conflict" Reviews ---
conflict_reviews_df = merged_df[
    (merged_df['stars_x'] >= 4) &
    (merged_df['vader_sentiment_x'] == 'Negative')
]
print(f"Found {len(conflict_reviews_df['review_id'].unique())} high-conflict reviews to analyze.")

# --- 5. Find Positive/Negative Aspects in Each Review ---
print("Categorizing aspects...")
def get_emotion_polarity(emotions_str):
    emotions = str(emotions_str).split(', ')
    if any(em in NEGATIVE_EMOTIONS for em in emotions): return 'Negative'
    if any(em in POSITIVE_EMOTIONS for em in emotions): return 'Positive'
    return 'Neutral'

conflict_reviews_df.loc[:, 'polarity'] = conflict_reviews_df['sentence_top_emotions'].apply(get_emotion_polarity)
grouped = conflict_reviews_df.groupby('review_id')
all_conflict_pairs = Counter()

print("Finding conflict pairs...")
for review_id, group in tqdm(grouped):
    positive_aspects = set(group[group['polarity'] == 'Positive']['aspect'])
    negative_aspects = set(group[group['polarity'] == 'Negative']['aspect'])
    
    if positive_aspects and negative_aspects:
        pairs = list(product(positive_aspects, negative_aspects))
        # We still filter "self-conflict"
        standardized_pairs = [tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1]]
        all_conflict_pairs.update(standardized_pairs)

if not all_conflict_pairs:
    print("No conflict pairs were found. Exiting.")
    exit()

# --- 6. Create the Heatmap Matrix ---
print("Building conflict co-occurrence matrix...")
# Find all unique aspects involved in conflict
all_aspects = sorted(list(set([aspect for pair in all_conflict_pairs.keys() for aspect in pair])))

# Create an empty matrix (DataFrame) filled with zeros
heatmap_df = pd.DataFrame(0, index=all_aspects, columns=all_aspects, dtype=int)

# Fill the matrix with our conflict counts
for (aspect1, aspect2), count in all_conflict_pairs.items():
    heatmap_df.loc[aspect1, aspect2] = count
    heatmap_df.loc[aspect2, aspect1] = count # Make it symmetrical

# --- 7. Plot the Heatmap ---
print("Generating heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(
    heatmap_df,
    annot=True,     # Show the numbers in the cells
    fmt='d',        # Format as integers
    cmap="Reds",    # Use a red color scale
    linewidths=.5   # Add lines between cells
)
plt.title("Conflict Co-occurrence Heatmap for High-Rated Negative Reviews", fontsize=16)
plt.xlabel("Negative Aspect", fontsize=12)
plt.ylabel("Positive Aspect", fontsize=12)
plt.tight_layout()

output_filename = "conflict_heatmap.png"
plt.savefig(output_filename)

print(f"\n✅ Success! Heatmap saved to {output_filename}")
print("\nAnalysis Complete.")