import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import Counter
import numpy as np

# Set visualization style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("--- Starting Comprehensive ABSA Interpretation (V2.1 - Fix) ---")

# --- 1. Load Data ---
print("Loading data...")
try:
    aspect_df = pd.read_csv("aspect_level_emotion_analysis.csv")
    review_df = pd.read_csv("medical_sentiment_mismatch.csv")
    
    if 'review_id' in review_df.columns:
        review_df = review_df.drop(columns=['review_id'])
    review_df = review_df.reset_index().rename(columns={'index': 'review_id'})
    
    # This merge creates 'stars_x' (from aspect_df) and 'stars_y' (from review_df)
    full_df = pd.merge(aspect_df, review_df, on='review_id', how='left')
    
except FileNotFoundError:
    print("Error: Required CSV files not found. Please run the full pipeline first.")
    exit()

# --- Helper: Emotion Polarity ---
POSITIVE_EMOTIONS = {'joy', 'gratitude', 'love', 'admiration', 'approval', 'caring', 'excitement', 'optimism', 'relief', 'pride'}
NEGATIVE_EMOTIONS = {'anger', 'disgust', 'fear', 'sadness', 'disappointment', 'annoyance', 'remorse', 'grief', 'embarrassment', 'nervousness'}

def get_emotion_polarity(emotions_str):
    emotions = str(emotions_str).split(', ')
    if any(em in NEGATIVE_EMOTIONS for em in emotions): return 'Negative'
    if any(em in POSITIVE_EMOTIONS for em in emotions): return 'Positive'
    return 'Neutral'

full_df['aspect_polarity'] = full_df['sentence_top_emotions'].apply(get_emotion_polarity)

# =========================================
# 📊 Plot 1: The "Intensity Gap" Heatmap
# =========================================
print("Generating Plot 1: Intensity Gap Heatmap...")
plt.figure(figsize=(10, 8))
# We use review_df here as it's the un-merged source for this plot
heatmap_data = pd.crosstab(review_df['stars'], review_df['vader_sentiment'])
heatmap_data = heatmap_data[['Negative', 'Neutral', 'Positive']].sort_index(ascending=False)
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Reviews'})
plt.title("The 'Intensity Gap': Star Ratings vs. Text Sentiment")
plt.ylabel("Star Rating")
plt.xlabel("VADER Text Sentiment")
plt.tight_layout()
plt.savefig("plot1_intensity_gap_heatmap.png")
plt.close()

# =========================================
# 📊 Plot 2: Top Negative Aspects in 5-Star Reviews
# =========================================
print("Generating Plot 2: Hidden Dissatisfaction...")
#
# ❗❗❗ THE FIX IS HERE ❗❗❗
# Using 'stars_x' (from the merged full_df)
#
hidden_neg_df = full_df[(full_df['stars_x'] == 5) & (full_df['aspect_polarity'] == 'Negative')]
top_neg_aspects = hidden_neg_df['aspect'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_neg_aspects.values, y=top_neg_aspects.index, palette='Reds_r')
plt.title("Hidden Dissatisfaction: Top Negative Aspects in 5-Star Reviews")
plt.xlabel("Number of Negative Mentions")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig("plot2_hidden_dissatisfaction_5star.png")
plt.close()

# =========================================
# 📊 Plot 3: Top Positive Aspects in 1-Star Reviews
# =========================================
print("Generating Plot 3: Silver Linings...")
#
# ❗❗❗ THE FIX IS HERE ❗❗❗
# Using 'stars_x' (from the merged full_df)
#
silver_lining_df = full_df[(full_df['stars_x'] == 1) & (full_df['aspect_polarity'] == 'Positive')]
top_pos_aspects = silver_lining_df['aspect'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_pos_aspects.values, y=top_pos_aspects.index, palette='Greens_r')
plt.title("Silver Linings: Top Positive Aspects in 1-Star Reviews")
plt.xlabel("Number of Positive Mentions")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig("plot3_silver_linings_1star.png")
plt.close()

# =========================================
# 📊 Plot 4: Emotion Distribution by Top Aspects
# =========================================
print("Generating Plot 4: Emotion Distribution...")
top_5_aspects = full_df['aspect'].value_counts().head(5).index
aspect_emotion = full_df[full_df['aspect'].isin(top_5_aspects)]

aspect_emotion.loc[:, 'emotion'] = aspect_emotion['sentence_top_emotions'].str.split(', ')
exploded = aspect_emotion.explode('emotion')
exploded.loc[:, 'emotion'] = exploded['emotion'].str.strip()

top_emotions = exploded['emotion'].value_counts().head(8).index
exploded = exploded[exploded['emotion'].isin(top_emotions)]

plt.figure(figsize=(14, 8))
sns.countplot(data=exploded, x='aspect', hue='emotion', hue_order=top_emotions, palette='Spectral')
plt.title("Emotion Distribution for Top 5 Aspects")
plt.ylabel("Count of Emotion Mentions")
plt.xlabel("Aspect")
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plot4_aspect_emotion_distribution.png")
plt.close()

# =========================================
# 📊 Plot 5: The "Conflict Archetypes" (Polished)
# =========================================
print("Generating Plot 5: Conflict Archetypes...")
#
# ❗❗❗ THE FIX IS HERE ❗❗❗
# Using 'stars_x' and 'vader_sentiment_x' (from the merged full_df)
#
conflict_reviews = full_df[(full_df['stars_x'] >= 4) & (full_df['vader_sentiment_x'] == 'Negative')]
grouped = conflict_reviews.groupby('review_id')
conflict_pairs = Counter()
for _, group in grouped:
    pos = set(group[group['aspect_polarity'] == 'Positive']['aspect'])
    neg = set(group[group['aspect_polarity'] == 'Negative']['aspect'])
    if pos and neg:
        for pair in product(pos, neg):
            if pair[0] != pair[1]:
                conflict_pairs.update([tuple(sorted(pair))])

top_archetypes = conflict_pairs.most_common(15)
arch_df = pd.DataFrame(top_archetypes, columns=['pair', 'count'])
arch_df['label'] = arch_df['pair'].apply(lambda x: f"{x[0]} ↔ {x[1]}")

plt.figure(figsize=(12, 8))
sns.barplot(data=arch_df, x='count', y='label', palette='magma')
plt.title("Top 15 'Emotional Conflict' Archetypes in High-Rated Reviews")
plt.xlabel("Frequency of Conflict Pattern")
plt.ylabel("Conflict Pair (Positive Aspect ↔ Negative Aspect)")
plt.tight_layout()
plt.savefig("plot5_conflict_archetypes.png")
plt.close()

# =========================================
# 📊 Plot 6: Net Sentiment by Aspect
# =========================================
print("Generating Plot 6: Net Sentiment...")
aspect_sentiment = full_df.groupby('aspect')['aspect_polarity'].value_counts().unstack(fill_value=0)
aspect_sentiment['total'] = aspect_sentiment.sum(axis=1)
pos_col = aspect_sentiment['Positive'] if 'Positive' in aspect_sentiment else 0
neg_col = aspect_sentiment['Negative'] if 'Negative' in aspect_sentiment else 0
aspect_sentiment['net_score'] = (pos_col - neg_col) / aspect_sentiment['total']

sorted_sentiment = aspect_sentiment.sort_values('net_score', ascending=False)
top_and_bottom = pd.concat([sorted_sentiment.head(5), sorted_sentiment.tail(5)])

plt.figure(figsize=(12, 8))
colors = ['green' if x > 0 else 'red' for x in top_and_bottom['net_score']]
sns.barplot(x=top_and_bottom['net_score'], y=top_and_bottom.index, palette=colors)
plt.title("Net Sentiment Score by Aspect (Top 5 Best vs. Top 5 Worst)")
plt.xlabel("Net Sentiment Score (-1.0 to +1.0)")
plt.ylabel("Aspect")
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig("plot6_net_sentiment_by_aspect.png")
plt.close()

print("\n✅ Success! All 6 interpretation plots have been saved.")