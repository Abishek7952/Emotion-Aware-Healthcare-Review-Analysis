import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import Counter
from tqdm import tqdm

print("--- Starting Conflict Pair Analysis (V5 - Final Filter) ---")

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
    if any(em in NEGATIVE_EMOTIONS for em in emotions):
        return 'Negative'
    if any(em in POSITIVE_EMOTIONS for em in emotions):
        return 'Positive'
    return 'Neutral'

conflict_reviews_df.loc[:, 'polarity'] = conflict_reviews_df['sentence_top_emotions'].apply(get_emotion_polarity)

grouped = conflict_reviews_df.groupby('review_id')
all_conflict_pairs = Counter()

print("Finding conflict pairs (this may take a moment)...")
for review_id, group in tqdm(grouped):
    positive_aspects = set(group[group['polarity'] == 'Positive']['aspect'])
    negative_aspects = set(group[group['polarity'] == 'Negative']['aspect'])
    
    if positive_aspects and negative_aspects:
        pairs = list(product(positive_aspects, negative_aspects))
        
        #
        # ❗❗❗ THIS IS THE FINAL FIX ❗❗❗
        # We only count pairs where the aspects are DIFFERENT.
        # This filters out ('Doctor', 'Doctor') and keeps ('Doctor', 'Billing').
        #
        standardized_pairs = []
        for pair in pairs:
            if pair[0] != pair[1]: # This is the new rule!
                standardized_pairs.append(tuple(sorted(pair)))

        all_conflict_pairs.update(standardized_pairs)

print("\n--- 🏆 Top 20 Conflict Archetypes ---")
if not all_conflict_pairs:
    print("No conflict pairs were found in the dataset.")
else:
    top_20_pairs = all_conflict_pairs.most_common(20)
    
    for (aspect1, aspect2), count in top_20_pairs:
        print(f"   {count:>5}x : '{aspect1}' <---vs---> '{aspect2}'")

    # --- 6. Plot the Results ---
    print("\nGenerating plot...")
    plot_data = pd.DataFrame(top_20_pairs, columns=['pair', 'count'])
    plot_data['pair_label'] = plot_data['pair'].apply(lambda x: f"'{x[0]}' <--> '{x[1]}'")
    
    plt.figure(figsize=(10, 12))
    sns.barplot(
        x='count',
        y='pair_label',
        data=plot_data,
        palette="Reds_r"
    )
    plt.title("Top 20 'Conflict Archetypes' in High-Rated Negative Reviews", fontsize=16)
    plt.xlabel("Number of Occurrences", fontsize=12)
    plt.ylabel("Aspect Conflict Pair (Positive <--> Negative)", fontsize=12)
    plt.tight_layout()
    
    output_filename = "conflict_archetypes_plot.png"
    plt.savefig(output_filename)
    
    print(f"\n✅ Success! Plot saved to {output_filename}")

print("\nAnalysis Complete.")