import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

print("Starting Step 3: Fuzzy Matching Yelp and CMS data (V3 - Print Fix)")

# --- 1. Load Processed Data ---
try:
    yelp_df = pd.read_csv("yelp_scores_agg.csv")
    cms_df = pd.read_csv("cms_scores_processed.csv")
except FileNotFoundError:
    print("Error: 'yelp_scores_agg.csv' or 'cms_scores_processed.csv' not found.")
    print("Please run the previous scripts first.")
    exit()

print(f"Loaded {len(yelp_df)} Yelp hospitals and {len(cms_df)} CMS hospitals.")

# --- 2. Helper Function for Matching ---
def find_best_match(yelp_name, yelp_state, cms_options):
    best_score = 0
    best_match_id = None
    
    state_options = cms_options[cms_options['State'] == yelp_state]
    if state_options.empty:
        return None, 0

    for _, cms_row in state_options.iterrows():
        # Uses the correct 'Facility Name' column
        score = fuzz.token_set_ratio(yelp_name, cms_row['Facility Name'])
        
        if score > best_score:
            best_score = score
            best_match_id = cms_row['Facility ID']
            
    # Set a confidence threshold for matching
    if best_score > 90:
        return best_match_id, best_score
    else:
        return None, 0

# --- 3. Iterate and Match ---
print("Matching hospitals... This will take some time.")
matches = []
for _, yelp_row in tqdm(yelp_df.iterrows(), total=yelp_df.shape[0]):
    match_id, score = find_best_match(
        yelp_row['yelp_name'], 
        yelp_row['yelp_state'], 
        cms_df
    )
    
    if match_id is not None:
        matches.append({
            'business_id': yelp_row['business_id'],
            'Facility ID': match_id,
            'match_score': score
        })

match_df = pd.DataFrame(matches)
print(f"Found {len(match_df)} high-confidence matches.")

# --- 4. Create Final Merged Dataset ---
final_df = pd.merge(yelp_df, match_df, on='business_id')
final_df = pd.merge(final_df, cms_df, on='Facility ID')

# --- 5. Save Output ---
output_file = "final_merged_scores.csv"
final_df.to_csv(output_file, index=False)
print(f"\n✅ Success! Final merged analysis file saved to: {output_file}")

# Uses the correct 'avg_emotion_only_rating' column for the preview
print(final_df[['yelp_name', 'Facility Name', 'avg_original_stars', 'avg_emotion_only_rating', 'hcahps_score']].head())