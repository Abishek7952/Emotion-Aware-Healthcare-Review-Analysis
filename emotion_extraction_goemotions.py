from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# ❗ Force GPU only
assert torch.cuda.is_available(), "🚫 CUDA is not available. Please run on a machine with a GPU."

# Set device to GPU
device = torch.device("cuda")

# Load tokenizer and model, move model to GPU
tokenizer = AutoTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")
model.to(device)
model.eval()

# Emotion labels (from GoEmotions)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Function to get top N emotions in batch
def get_top_emotions_batch(texts, top_n=3):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)

    top_emotions = []
    for prob in probs:
        top_indices = torch.topk(prob, top_n).indices.tolist()
        emotions = [emotion_labels[i] for i in top_indices if i < len(emotion_labels)]
        top_emotions.append(', '.join(emotions))
    return top_emotions

# Load dataset
df = pd.read_csv("medical_reviews.csv")
texts = df['text'].fillna('').tolist()

# Batch processing
batch_size = 128  # Try 64 or 128 if GPU has more free VRAM
results = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    emotions = get_top_emotions_batch(batch)
    results.extend(emotions)

# Save results
df['top_emotions'] = results
df.to_csv("emotion_analysis_output.csv", index=False)

# Preview
print(df[['text', 'top_emotions']].head())
