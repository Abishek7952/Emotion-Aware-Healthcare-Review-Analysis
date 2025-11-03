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

# ❗❗ --- THE NEW HIERARCHICAL "BRAIN" --- ❗❗
# We search for "Entities" first. If we find one, we stop.
# This prevents a "rude doctor" from being tagged as both 'Doctor' and 'Communication'.

# Priority 1: The "Who" (Entities)
ENTITY_KEYWORDS = {
    'Doctor': ['doctor', 'dr', 'physician', 'surgeon', 'specialist', 'md'],
    'Staff': ['staff', 'receptionist', 'front desk', 'nurse', 'nurses', 'admin', 'assistant'],
    'Billing': ['bill', 'billing', 'cost', 'price', 'insurance', 'charge', 'charged', 'payment', 'expensive']
}

# Priority 2: The "What/Where" (Topics)
TOPIC_KEYWORDS = {
    'Wait Time': ['wait', 'waiting', 'long', 'hour', 'hours', 'appointment', 'schedule', 'punctual'],
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

# --- Aspect Extraction Function (NEW HIERARCHY) ---
def extract_keyword_aspects(text):
    doc = nlp(text)
    aspect_sentences = {} # We'll build this dynamically
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        found_aspect = None
        
        # Priority 1: Check for Entities
        for aspect_name, keywords in ENTITY_KEYWORDS.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', sent_text) for kw in keywords):
                found_aspect = aspect_name
                break # Found the most important aspect, stop checking
        
        # Priority 2: Check for Topics *only if* no Entity was found
        if not found_aspect:
            for aspect_name, keywords in TOPIC_KEYWORDS.items():
                if any(re.search(r'\b' + re.escape(kw) + r'\b', sent_text) for kw in keywords):
                    found_aspect = aspect_name
                    break # Found the topic, stop checking
        
        # If we found any aspect, add it to our list
        if found_aspect:
            if found_aspect not in aspect_sentences:
                aspect_sentences[found_aspect] = []
            aspect_sentences[found_aspect].append(sent.text)
            
    return aspect_sentences


# --- Main Processing Loop (NEW) ---
print("Processing mismatched reviews for Aspect-Level Emotion...")
results_list = []

for index, row in tqdm(mismatch_df.iterrows(), total=mismatch_df.shape[0]):
    text = str(row['text'])
    aspect_sentences_map = extract_keyword_aspects(text)
    
    all_sentences = []
    all_aspects = []
    for aspect_name, sentences in aspect_sentences_map.items():
        for sent in sentences:
            all_sentences.append(sent)
            all_aspects.append(aspect_name)
    
    if not all_sentences:
        continue

    try:
        top_emotions = get_top_emotions_batch(all_sentences, top_n=3)
    except Exception as e:
        print(f"Error processing batch: {e}")
        continue
    
    for i, aspect_name in enumerate(all_aspects):
        results_list.append({
            'review_id': row['original_index'], 
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
else:
    print("\nNo aspects were extracted from the mismatched reviews.")