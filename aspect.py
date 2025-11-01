# emotion_sentiment_topic_gpu.py
import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from bertopic import BERTopic

# ---------- CONFIG ----------
INPUT_CSV = r"D:\NLP\medical_sentiment_mismatch.csv"
OUTPUT_CSV = r"D:\NLP\aspect_emotion_rating_gpu.csv"
TOPIC_CSV = r"D:\NLP\emotion_topics_summary.csv"
GOEMO_MODEL = "monologg/bert-base-cased-goemotions-original"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # tune (16–64 works best for 12GB VRAM)
# ----------------------------

print(f"🟢 Using device: {DEVICE}")

# Load Data
df = pd.read_csv(INPUT_CSV)
df['review'] = df['text'].astype(str)

# Load NLP and Models
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
vader = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained(GOEMO_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(GOEMO_MODEL).to(DEVICE)
model.eval()

EMOTION_LABELS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

POLARITY_MAP = {
    'joy': 1.0, 'love': 1.0, 'admiration': 0.8, 'approval': 0.6, 'gratitude': 0.8,
    'relief': 0.6, 'optimism': 0.5, 'excitement': 0.7,
    'anger': -1.0, 'disgust': -1.0, 'sadness': -0.9, 'fear': -0.8,
    'annoyance': -0.6, 'disappointment': -0.7, 'remorse': -0.6,
    'nervousness': -0.4, 'embarrassment': -0.3
}

# --- Helper Functions ---

def get_sentiment_score(text):
    return vader.polarity_scores(text)['compound']

def get_emotion_probs_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
    return probs

def emotion_polarity_vector(probs):
    polarity_scores = []
    for prob in probs:
        score = sum(POLARITY_MAP.get(label, 0) * p for label, p in zip(EMOTION_LABELS, prob))
        polarity_scores.append(score)
    return np.array(polarity_scores)

# --- Emotion-Aware Rating ---
final_results = []
print("🔄 Running emotion-aware sentiment adjustment on GPU...")

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_texts = df['review'][i:i+BATCH_SIZE].tolist()
    batch_ratings = df['stars'][i:i+BATCH_SIZE].tolist() if 'stars' in df.columns else [3.0] * len(batch_texts)

    sentiments = [get_sentiment_score(t) for t in batch_texts]
    emo_probs = get_emotion_probs_batch(batch_texts)
    emo_pols = emotion_polarity_vector(emo_probs)

    alpha = 1.5
    weighted_scores = 0.6 * np.array(sentiments) + 0.4 * emo_pols
    adjusted_ratings = np.clip(np.array(batch_ratings) + weighted_scores * alpha, 1.0, 5.0)

    for text, orig, adj, sent, emo in zip(batch_texts, batch_ratings, adjusted_ratings, sentiments, emo_pols):
        final_results.append({
            'text': text,
            'original_rating': orig,
            'vader_sentiment': sent,
            'emotion_polarity': emo,
            'adjusted_rating': round(float(adj), 2)
        })

# Save emotion results
out_df = pd.DataFrame(final_results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Emotion-aware ratings saved to: {OUTPUT_CSV}")

# --- Topic Modeling (BERTopic) ---
print("🔍 Running BERTopic clustering for thematic insights...")

texts = out_df['text'].astype(str).tolist()
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)

topics, probs = topic_model.fit_transform(texts)
out_df['topic'] = topics
out_df['topic_confidence'] = [max(p) if isinstance(p, list) else None for p in probs]

topic_info = topic_model.get_topic_info()
topic_summary = out_df.groupby('topic')['adjusted_rating'].mean().reset_index()
topic_summary = topic_summary.merge(topic_info, on='topic')

topic_summary.to_csv(TOPIC_CSV, index=False)
print(f"🏁 Topic clusters with emotion-aware ratings saved to: {TOPIC_CSV}")

# Optional visualization (interactive HTML)
try:
    topic_model.visualize_topics().write_html(r"D:\NLP\emotion_topics_viz.html")
    print("📊 Visualization saved: emotion_topics_viz.html")
except Exception as e:
    print(f"⚠️ Visualization skipped: {e}")
