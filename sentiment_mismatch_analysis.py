import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Setup VADER
analyzer = SentimentIntensityAnalyzer()
tqdm.pandas()

# Load data
print("Loading medical_reviews.csv...")
try:
    df = pd.read_csv("medical_reviews.csv")
except FileNotFoundError:
    print("Error: medical_reviews.csv not found. Please run filter medical reviews.py first.")
    exit()

df['text'] = df['text'].astype(str)

# Function to get VADER label and compound score
def get_vader_scores(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    
    # Return both the label AND the raw compound score
    return label, compound

# Apply the function
print("Running VADER analysis...")
# This creates two new columns from the two outputs
df[['vader_sentiment', 'vader_compound']] = df['text'].progress_apply(
    lambda text: pd.Series(get_vader_scores(text))
)

# Function to get rating label
def get_rating_label(stars):
    if stars >= 4:
        return 'Positive'
    elif stars < 3:
        return 'Negative'
    else:
        return 'Neutral'

df['rating_sentiment'] = df['stars'].apply(get_rating_label)

# Find mismatches
df['sentiment_mismatch'] = df['vader_sentiment'] != df['rating_sentiment']

# Save output
output_file = "medical_sentiment_mismatch.csv"
# We save with index=False to keep the file clean
df.to_csv(output_file, index=False)

print(f"\n✅ Success! Sentiment analysis complete.")
print(f"New file saved to: {output_file}")