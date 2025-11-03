import pandas as pd

print("Starting: Preparing CMS Data...")

# --- 1. Load Downloaded CMS Files ---
try:
    # Reading the exact filenames from your screenshot
    print("Loading Hospital_General_Information.csv...")
    info_df = pd.read_csv("Hospital_General_Information.csv", encoding='latin1')
    
    print("Loading Patient_Survey_(HCAHPS)_-_Hospital.csv...")
    scores_df = pd.read_csv("Patient_Survey_(HCAHPS)_-_Hospital.csv", encoding='latin1', low_memory=False) 
except FileNotFoundError:
    print("Error: Could not find 'Hospital_General_Information.csv' or 'Patient_Survey_(HCAHPS)_-_Hospital.csv'")
    exit()

print("Files loaded. Processing...")

# --- 2. Filter for the "Summary Star Rating" ---
# From our previous debugging, we know the correct ID is 'H_STAR_RATING'
hcahps_target = scores_df[
    scores_df['HCAHPS Measure ID'] == 'H_STAR_RATING'
]

if hcahps_target.empty:
    print("FATAL ERROR: Filtered for 'H_STAR_RATING' but found no data.")
    exit()

hcahps_target = hcahps_target.copy() 

# From our debugging, we know the correct column name is 'Patient Survey Star Rating'
hcahps_target['hcahps_score'] = pd.to_numeric(
    hcahps_target['Patient Survey Star Rating'], 
    errors='coerce'
)

# --- 3. Clean and Select ---
hcahps_cleaned = hcahps_target.dropna(subset=['hcahps_score'])
hcahps_final = hcahps_cleaned[['Facility ID', 'hcahps_score']]

# --- 4. Merge Info and Scores ---
# This merges the hospital info (name, address) with its summary score
cms_df = pd.merge(
    info_df, 
    hcahps_final, 
    on='Facility ID', 
    how='inner'
)

# --- 5. Save Output ---
output_file = "cms_scores_processed.csv"
cms_df.to_csv(output_file, index=False)
print(f"\n✅ Success! Processed CMS scores saved to: {output_file}")

# Use the correct column names for the print preview
print(cms_df[['Facility ID', 'Facility Name', 'Address', 'City/Town', 'State', 'hcahps_score']].head())