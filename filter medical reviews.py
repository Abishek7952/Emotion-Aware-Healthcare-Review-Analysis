import pandas as pd
import json
import os

# === File Paths ===
business_path = r"D:\NLP\data\yelp_academic_dataset_business.json"
review_path = r"D:\NLP\data\yelp_academic_dataset_review.json"
output_path = r"D:\NLP\medical_reviews.csv"

# === Step 1: Load business data and filter for medical-related categories ===
print("Loading business data...")
businesses = []
with open(business_path, encoding='utf8') as f:
    for line in f:
        businesses.append(json.loads(line))
business_df = pd.DataFrame(businesses)

# Define medical keywords
medical_keywords = ["Hospital", "Doctor", "Dentist", "Clinic", "Medical", "Health", "Urgent Care"]
mask = business_df['categories'].fillna('').str.contains('|'.join(medical_keywords), case=False)
medical_businesses = business_df[mask]
print(f"Filtered medical businesses: {medical_businesses.shape}")

# Create a set of medical business_ids for fast lookup
medical_ids = set(medical_businesses['business_id'])

# === Step 2: Stream and filter review data ===
print("Filtering medical reviews...")
filtered_reviews = []
with open(review_path, encoding='utf8') as f:
    for line in f:
        review = json.loads(line)
        if review['business_id'] in medical_ids:
            filtered_reviews.append(review)

print(f"Total medical reviews found: {len(filtered_reviews)}")

# Convert to DataFrame
review_df = pd.DataFrame(filtered_reviews)

# === Step 3: Save to CSV ===
review_df.to_csv(output_path, index=False)
print(f"Filtered medical reviews saved to: {output_path}")
