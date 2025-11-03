import pandas as pd
import json

print("Starting Step 1: Aggregating Yelp Scores (V8 - Using New Emotion-Only Rating)")

# --- 1. Load Data ---
print("Loading all review data...")
try:
    # We will use the 'emotion_only_rating' column from this file
    ratings_df = pd.read_csv("final_absa_adjusted_ratings.csv") 
except FileNotFoundError:
    print("Error: final_absa_adjusted_ratings.csv not found.")
    print("Please run calculate_absa_ear.py (V2) first.")
    exit()

print("Loading Yelp business.json...")
try:
    business_path = r"D:\NLP\data\yelp_academic_dataset_business.json" 
    businesses = []
    with open(business_path, encoding='utf8') as f:
        for line in f:
            businesses.append(json.loads(line))
    business_df = pd.DataFrame(businesses)
except FileNotFoundError:
    print(f"Error: Yelp Business JSON not found.")
    exit()

# --- 2. Merge and Aggregate ---
print("Merging data sources...")
full_yelp_data = pd.merge(ratings_df, business_df, on='business_id', how='left')

print("Aggregating scores by hospital...")
agg_df = full_yelp_data.groupby('business_id').agg(
    yelp_name=('name', 'first'),
    yelp_address=('address', 'first'),
    yelp_city=('city', 'first'),
    yelp_state=('state', 'first'),
    #
    # ❗ Using 'stars_x' to prevent the KeyError
    #
    yelp_review_count=('stars_x', 'count'),       
    avg_original_stars=('stars_x', 'mean'),
    #
    # ❗❗❗ THIS IS THE KEY CHANGE ❗❗❗
    # We are now averaging your new, non-clone score.
    #
    avg_emotion_only_rating=('emotion_only_rating', 'mean')
).reset_index()

# Filter out hospitals with few reviews
agg_df = agg_df[agg_df['yelp_review_count'] >= 5]

# --- 3. Save Output ---
output_file = "yelp_scores_agg.csv"
agg_df.to_csv(output_file, index=False)
print(f"\n✅ Success! Aggregated Yelp scores (V8) with new rating saved to: {output_file}")
print(agg_df[['yelp_name', 'avg_original_stars', 'avg_emotion_only_rating']].head())