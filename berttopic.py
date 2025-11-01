from bertopic import BERTopic
from umap import UMAP
import pandas as pd

# Load your processed reviews
df = pd.read_csv(r"D:\NLP\aspect_emotion_rating_gpu.csv")
texts = df['text'].astype(str).tolist()  # or any column you want to cluster

# UMAP for dimensionality reduction (low memory)
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,   # lower dims = less RAM
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# BERTopic model
topic_model = BERTopic(
    language="english",
    umap_model=umap_model,
    calculate_probabilities=True,
    verbose=True,
    low_memory=True   # key for large datasets
)

# Fit and transform
topics, probs = topic_model.fit_transform(texts)

# Save topics
df['topic'] = topics
df.to_csv(r"D:\NLP\aspect_emotion_bertopic.csv", index=False)

print("✅ BERTopic clustering completed and saved.")
