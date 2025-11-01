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
BATCH_SIZE = 32 # You can tune this
# ----------------------------

print(f"🟢 Using device: {DEVICE}")

# Load Data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: {INPUT_CSV} not found.")
    exit()

# Filter for mismatches only. This is key for your research.
mismatch_df = df[df['sentiment_mismatch'] == True].copy()
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

# --- Model Functions (from your emotion_extraction_goemotions.py) ---

def get_top_emotions_batch(texts, top_n=3):
    """
    Analyzes a batch of texts and returns the top N emotions for each.
    """
    if not texts:
        return []
        
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

# --- Aspect Extraction Function (The "Novel" Part) ---

def extract_aspects_with_context(doc):
    """
    Extracts aspects (nouns) and their associated opinions (adjectives).
    This is a common rule-based method for ABSA.
    """
    aspects = []
    for token in doc:
        # Rule 1: Adjective modifying a noun (e.g., "GREAT doctor")
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            aspect = token.head.text.lower()
            opinion = token.text.lower()
            aspects.append((aspect, opinion, token.sent.text))
            
        # Rule 2: Noun subject of an adjective (e.g., "doctor was GREAT")
        elif token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "ADJ":
            aspect = token.text.lower()
            opinion = token.head.text.lower()
            aspects.append((aspect, opinion, token.sent.text))

    # Deduplicate based on sentence
    seen_sentences = set()
    unique_aspects = []
    for aspect, opinion, sentence in aspects:
        if sentence not in seen_sentences:
            unique_aspects.append((aspect, opinion, sentence))
            seen_sentences.add(sentence)
            
    return unique_aspects

# --- Main Processing Loop ---

print("Processing mismatched reviews for Aspect-Level Emotion...")

# This will store all our findings
results_list = []

# We process the dataframe row by row (slow, but clear)
# For speed, you'd batch the emotion analysis, but this is more robust.
for index, row in tqdm(mismatch_df.iterrows(), total=mismatch_df.shape[0]):
    text = str(row['text'])
    doc = nlp(text)
    
    # Get (aspect, opinion, sentence) tuples
    extracted_aspects = extract_aspects_with_context(doc)
    
    if not extracted_aspects:
        continue

    # Prepare sentences for batch emotion analysis
    sentences_to_analyze = [sent for _, _, sent in extracted_aspects]
    
    # Get emotions for this batch of sentences
    try:
        top_emotions = get_top_emotions_batch(sentences_to_analyze, top_n=3)
    except Exception as e:
        print(f"Error processing batch: {e}")
        continue
    
    # Combine results
    for i, (aspect, opinion, sentence) in enumerate(extracted_aspects):
        results_list.append({
            'review_id': index, # Use index as a unique ID
            'stars': row['stars'],
            'vader_sentiment': row['vader_sentiment'],
            'rating_sentiment': row['rating_sentiment'],
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
    print("\nSample Output:")
    print(final_aspect_df.head())
else:
    print("\nNo aspects were extracted from the mismatched reviews.")