import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---------- CONFIG ----------
INPUT_CSV = "medical_sentiment_mismatch.csv"
OUTPUT_CSV = "aspect_level_emotion_analysis.csv"
GOEMO_MODEL = "monologg/bert-base-cased-goemotions-original"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
# ----------------------------

print(f"🟢 Using device: {DEVICE}")

# Load Data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found. Run sentiment_mismatch_analysis.py first.")
    exit()

mismatch_df = df[df['sentiment_mismatch'] == True].copy()
#
# ❗❗❗ THIS IS THE BUG FIX ❗❗❗
# We reset the index to get the 'original_index' (e.g., 25, 35, 53)
#
mismatch_df = mismatch_df.reset_index().rename(columns={'index': 'original_index'})

print(f"Total reviews: {len(df)}, Mismatched reviews to analyze: {len(mismatch_df)}")

if mismatch_df.empty:
    print("No mismatched reviews found. Exiting.")
    exit()

# Load NLP Models
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

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

# --- Aspect Extraction Function ---
def extract_aspects_with_context(doc):
    aspects = []
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            aspect = token.head.text.lower()
            opinion = token.text.lower()
            aspects.append((aspect, opinion, token.sent.text))
        elif token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "ADJ":
            aspect = token.text.lower()
            opinion = token.head.text.lower()
            aspects.append((aspect, opinion, token.sent.text))
    seen_sentences = set()
    unique_aspects = []
    for aspect, opinion, sentence in aspects:
        if sentence not in seen_sentences:
            unique_aspects.append((aspect, opinion, sentence))
            seen_sentences.add(sentence)
    return unique_aspects

# --- Main Processing Loop ---
print("Processing mismatched reviews for Aspect-Level Emotion...")
results_list = []

for index, row in tqdm(mismatch_df.iterrows(), total=mismatch_df.shape[0]):
    text = str(row['text'])
    doc = nlp(text)
    extracted_aspects = extract_aspects_with_context(doc)
    if not extracted_aspects: continue

    sentences_to_analyze = [sent for _, _, sent in extracted_aspects]
    try:
        top_emotions = get_top_emotions_batch(sentences_to_analyze, top_n=3)
    except Exception as e:
        print(f"Error processing batch: {e}")
        continue
    
    for i, (aspect, opinion, sentence) in enumerate(extracted_aspects):
        results_list.append({
            #
            # ❗❗❗ THIS IS THE BUG FIX ❗❗❗
            # We are saving the 'original_index' as the 'review_id'
            #
            'review_id': row['original_index'], 
            'stars': row['stars'],
            'vader_sentiment': row['vader_sentiment'],
            'aspect': aspect,
            'opinion': opinion,
            'sentence': sentence,
            'sentence_top_emotions': top_emotions[i]
        })

# --- Save Final Output ---
if results_list:
    final_aspect_df = pd.DataFrame(results_list)
    final_aspect_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Success! Aspect-level emotion analysis complete.")
    print(f"Output saved to: {OUTPUT_CSV}")
else:
    print("\nNo aspects were extracted from the mismatched reviews.")