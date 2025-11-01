import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --- Load New Data Files ---
try:
    aspect_df = pd.read_csv("aspect_level_emotion_analysis.csv")
    ratings_df = pd.read_csv("final_absa_adjusted_ratings.csv")
except FileNotFoundError:
    print("Error: Could not find 'aspect_level_emotion_analysis.csv' or 'final_absa_adjusted_ratings.csv'")
    print("Please make sure both files are in the same directory.")
    exit()

print("Data loaded. Generating new visualizations...")

# ===============================
# 📊 1. Boxplot: Original vs NEW ABSA-Adjusted Rating
# ===============================
plt.figure()
# We only want to plot the reviews that actually *changed*
changed_df = ratings_df[ratings_df['stars'] != ratings_df['absa_adjusted_rating']]
sns.boxplot(data=changed_df[['stars', 'absa_adjusted_rating']])
plt.title("📊 Distribution: Original vs NEW ABSA-Adjusted Ratings (Mismatched Only)")
plt.ylabel("Rating")
plt.xticks([0, 1], ['Original Rating', 'ABSA-Adjusted Rating'])
plt.tight_layout()
plt.savefig("1_absa_boxplot_original_vs_adjusted.png")
print("Saved: 1_absa_boxplot_original_vs_adjusted.png")

# ===============================
# 🔁 2. Rating Change Distribution (New ABSA Rating)
# ===============================
ratings_df['rating_change'] = ratings_df['absa_adjusted_rating'] - ratings_df['stars']
ratings_df['rating_change_status'] = ratings_df['rating_change'].apply(
    lambda x: 'Increased' if x > 0 else 'Decreased' if x < 0 else 'Same'
)

plt.figure()
# We only care about the mismatched reviews that had an adjustment
sns.countplot(
    x='rating_change_status', 
    data=ratings_df[ratings_df['sentiment_mismatch'] == True], 
    order=['Increased', 'Decreased', 'Same'],
    palette="Set2"
)
plt.title("🔁 Rating Change Summary (Mismatched Reviews Only)")
plt.xlabel("Change Type (Original vs. ABSA-Adjusted)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("2_absa_rating_change_summary.png")
print("Saved: 2_absa_rating_change_summary.png")

# ===============================
# 🌟 3. (NOVEL) Top Aspects in 5-Star Negative Mismatches
# ===============================
# Find 5-star reviews that VADER labeled Negative (a clear mismatch)
mismatch_5star_ids = ratings_df[
    (ratings_df['stars'] >= 4) & (ratings_df['vader_sentiment'] == 'Negative')
].index

# Filter aspect_df for these specific reviews
mismatch_5star_aspects = aspect_df[aspect_df['review_id'].isin(mismatch_5star_ids)]

# Explode emotions
mismatch_5star_aspects['emotion'] = mismatch_5star_aspects['sentence_top_emotions'].str.split(', ')
exploded_5star = mismatch_5star_aspects.explode('emotion')

# Find aspects linked to strong negative emotions
neg_emo_aspects = exploded_5star[
    exploded_5star['emotion'].isin(['anger', 'disgust', 'sadness', 'fear', 'disappointment', 'annoyance'])
]
top_neg_aspects = neg_emo_aspects['aspect'].value_counts().head(15)

plt.figure(figsize=(10, 8))
sns.barplot(y=top_neg_aspects.index, x=top_neg_aspects.values, palette="Reds_r")
plt.title("😡 Top 15 Aspects Linked to NEGATIVE Emotions in 4-5 Star Reviews")
plt.xlabel("Frequency of Aspect")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig("3_absa_top_neg_aspects_in_high_ratings.png")
print("Saved: 3_absa_top_neg_aspects_in_high_ratings.png")

# ===============================
# 🌟 4. (NOVEL) Top Aspects in 1-Star Positive Mismatches
# ===============================
# Find 1/2-star reviews that VADER labeled Positive (a clear mismatch)
mismatch_1star_ids = ratings_df[
    (ratings_df['stars'] <= 2) & (ratings_df['vader_sentiment'] == 'Positive')
].index

# Filter aspect_df for these specific reviews
mismatch_1star_aspects = aspect_df[aspect_df['review_id'].isin(mismatch_1star_ids)]

# Explode emotions
mismatch_1star_aspects['emotion'] = mismatch_1star_aspects['sentence_top_emotions'].str.split(', ')
exploded_1star = mismatch_1star_aspects.explode('emotion')

# Find aspects linked to strong positive emotions
pos_emo_aspects = exploded_1star[
    exploded_1star['emotion'].isin(['joy', 'gratitude', 'admiration', 'approval', 'caring', 'love'])
]
top_pos_aspects = pos_emo_aspects['aspect'].value_counts().head(15)

plt.figure(figsize=(10, 8))
sns.barplot(y=top_pos_aspects.index, x=top_pos_aspects.values, palette="Greens_r")
plt.title("😊 Top 15 Aspects Linked to POSITIVE Emotions in 1-2 Star Reviews")
plt.xlabel("Frequency of Aspect")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig("4_absa_top_pos_aspects_in_low_ratings.png")
print("Saved: 4_absa_top_pos_aspects_in_low_ratings.png")

print("\n✅ All new visualizations saved! Check your project folder.")