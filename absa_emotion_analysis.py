import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re

# ---------- CONFIG ----------
INPUT_CSV = "medical_sentiment_mismatch.csv"
OUTPUT_CSV = "aspect_level_emotion_analysis.csv" # We will overwrite this
GOEMO_MODEL = "monologg/bert-base-cased-goemotions-original"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
# ----------------------------

# ❗❗ --- THIS IS THE NEW "BRAIN" --- ❗❗
# We define our domain-specific keywords.
# This is much smarter than the old grammar rules.
ASPECT_KEYWORDS = {
    'Doctor': ['doctor', 'dr', 'physician', 'surgeon', 'specialist', 'md'],
    'Staff': ['staff', 'receptionist', 'front desk', 'nurse', 'nurses', 'admin', 'assistant'],
    'Wait Time': ['wait', 'waiting', 'long', 'hour', 'hours', 'appointment', 'schedule', 'punctual'],
    'Billing': ['bill', 'billing', 'cost', 'price', 'insurance', 'charge', 'charged', 'payment', 'expensive'],
    'Communication': ['explained', 'listened', 'answered', 'questions', 'communication', 'rude', 'friendly'],
    'Facility': ['facility', 'office', 'clinic', 'building', 'clean', 'dirty', 'parking', 'room']
}
# -------------------------------------

print(f"🟢 Using device: {DEVICE}")

# Load Data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found. Run sentiment_mismatch_analysis.py first.")
    exit()

mismatch_df = df[df['sentiment_mismatch'] == True].copy()
mismatch_df = mismatch_df.reset_index().rename(columns={'index': 'original_index'})
print(f"Total reviews: {len(df)}, Mismatched reviews to analyze: {len(mismatch_df)}")

# Load Models
print("Loading spaCy model (for sentence splitting)...")
# We only use spaCy to split text into sentences
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
nlp.add_pipe('sentencizer')

print(f"Loading GoEmotions model: {GOEMO_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(GOEMO_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(GOEMO_MODEL).to(DEVICE)
model.eval()

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# --- Model Functions ---
def get_top_emotions_batch(texts, top_n=3):
    if not texts: return []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)
    top_emotions_list = []
    for prob in probs:
        top_indices = torch.topk(prob, top_n).indices.tolist()
        emotions = [EMOTION_LABELS[i] for i in top_indices if i < len(EMOTION_LABELS)]
        top_emotions_list.append(', '.join(emotions))
    return top_emotions_list

# --- Aspect Extraction Function (NEW) ---
def extract_keyword_aspects(text):
    """
    Finds sentences that contain our keywords.
    Returns a dictionary of {Aspect: [list of sentences]}
    """
    doc = nlp(text)
    aspect_sentences = {key: [] for key in ASPECT_KEYWORDS}
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        found_aspect = False
        for aspect_name, keywords in ASPECT_KEYWORDS.items():
            # Use regex to find whole words
            if any(re.search(r'\b' + re.escape(kw) + r'\b', sent_text) for kw in keywords):
                aspect_sentences[aspect_name].append(sent.text)
                found_aspect = True
                # Optional: break after first match to avoid one sentence
                # being tagged as both 'Doctor' and 'Staff'
                break 
    return aspect_sentences


# --- Main Processing Loop (NEW) ---
print("Processing mismatched reviews for Aspect-Level Emotion...")
results_list = []

for index, row in tqdm(mismatch_df.iterrows(), total=mismatch_df.shape[0]):
    text = str(row['text'])
    
    # 1. Find all sentences related to our aspects
    aspect_sentences_map = extract_keyword_aspects(text)
    
    # 2. Batch-analyze all found sentences
    all_sentences = []
    all_aspects = []
    for aspect_name, sentences in aspect_sentences_map.items():
        for sent in sentences:
            all_sentences.append(sent)
            all_aspects.append(aspect_name)
    
    if not all_sentences:
        continue

    # 3. Get emotions for this review's sentences
    try:
        top_emotions = get_top_emotions_batch(all_sentences, top_n=3)
    except Exception as e:
        print(f"Error processing batch: {e}")
        continue
    
    # 4. Save the clean results
    for i, aspect_name in enumerate(all_aspects):
        results_list.append({
            'review_id': row['original_index'], # The correct, original ID
            'stars': row['stars'],
            'vader_sentiment': row['vader_sentiment'],
            'aspect': aspect_name, # The CLEAN aspect (e.g., "Doctor", "Staff")
            'sentence': all_sentences[i],
            'sentence_top_emotions': top_emotions[i]
        })

# --- Save Final Output ---
if results_list:
    final_aspect_df = pd.DataFrame(results_list)
    final_aspect_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Success! Aspect-level emotion analysis complete.")
    print(f"New, clean data saved to: {OUTPUT_CSV}")
    print("\nSample Output (Notice the clean 'aspect' column):")
    print(final_aspect_df.head())
else:
    print("\nNo aspects were extracted from the mismatched reviews.")