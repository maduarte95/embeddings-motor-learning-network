"""
Topic Modeling with BERTopic and Specter2
==========================================

DEPRECATED — use ``topic_modeling_new.py`` instead.
This script is kept for reference only. The active pipeline lives in
``topic_modeling_new.py`` (graph-fingerprint caching, embedding/topic-model
caches, and SPECTER2 embeddings obtained via ``embedding_loaders.embed_specter2``).
New work (embedding swap tests, topic-quality metrics) builds on that script and
on ``embedding_loaders.py``; this file is no longer maintained.

Loads the citation network graph, extracts paper titles and abstracts,
computes Specter2 embeddings, applies BERTopic to discover topics,
and adds the topic assignments as a new node attribute.

Outputs: Updated GraphML file with 'topic' attribute for each node.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel
import torch
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path 

DATA_DIR = Path("data")
GRAPHML_FILE = DATA_DIR / "citation_network_selected.graphml"

# ── Configuration ──────────────────────────────────────────────────────────
SPECTER2_MODEL = "allenai/specter2_base"  # or "allenai/specter2" for the original
OUTPUT_GRAPHML = DATA_DIR / "citation_network_with_topics.graphml"
MIN_TOPIC_SIZE = 15  # Minimum papers per topic
N_NEIGHBORS = 15     # UMAP parameter
N_COMPONENTS = 5     # UMAP dimensions
BATCH_SIZE = 32      # For embedding computation

print("=" * 80)
print("Topic Modeling with BERTopic and Specter2")
print("=" * 80)

# ── 1. Load Graph ─────────────────────────────────────────────────────────
print("\n[1/6] Loading graph...")
G = nx.read_graphml(GRAPHML_FILE)
print(f"  Loaded {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")

# ── 2. Extract Text Data ──────────────────────────────────────────────────
print("\n[2/6] Extracting titles and abstracts...")
documents = []
node_ids = []
valid_count = 0
missing_text = 0

for node, data in G.nodes(data=True):
    title = data.get('title', '').strip()
    abstract = data.get('abstract', '').strip()
    
    # Combine title and abstract
    if title or abstract:
        # Specter2 expects format: [TITLE] [SEP] [ABSTRACT]
        text = f"{title} [SEP] {abstract}" if abstract else title
        documents.append(text)
        node_ids.append(node)
        valid_count += 1
    else:
        missing_text += 1
        # Store empty string for nodes without text (will assign topic -1)
        documents.append("")
        node_ids.append(node)

print(f"  Papers with text: {valid_count:,}")
print(f"  Papers without text: {missing_text:,}")

# ── 3. Compute Specter2 Embeddings ────────────────────────────────────────
print("\n[3/6] Computing Specter2 embeddings...")
print(f"  Loading model: {SPECTER2_MODEL}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
model = AutoModel.from_pretrained(SPECTER2_MODEL).to(device)
model.eval()

def embed_batch(texts):
    """Embed a batch of texts using Specter2."""
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embeddings

# Compute embeddings in batches
embeddings_list = []
valid_docs = [doc for doc in documents if doc]  # Only embed non-empty documents
valid_indices = [i for i, doc in enumerate(documents) if doc]

print(f"  Embedding {len(valid_docs):,} documents in batches of {BATCH_SIZE}...")
for i in tqdm(range(0, len(valid_docs), BATCH_SIZE), desc="  Embedding"):
    batch = valid_docs[i:i + BATCH_SIZE]
    batch_embeddings = embed_batch(batch)
    embeddings_list.append(batch_embeddings)

# Concatenate all embeddings
valid_embeddings = np.vstack(embeddings_list)
print(f"  Embedding shape: {valid_embeddings.shape}")

# Create full embeddings array with zeros for missing documents
embeddings = np.zeros((len(documents), valid_embeddings.shape[1]))
embeddings[valid_indices] = valid_embeddings

# ── 4. Apply BERTopic ─────────────────────────────────────────────────────
print("\n[4/6] Applying BERTopic clustering...")

# Configure dimensionality reduction
umap_model = UMAP(
    n_neighbors=N_NEIGHBORS,
    n_components=N_COMPONENTS,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# Configure clustering
hdbscan_model = HDBSCAN(
    min_cluster_size=MIN_TOPIC_SIZE,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# Configure vectorizer for topic representation
vectorizer_model = CountVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2)
)

# Initialize BERTopic
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    verbose=True,
    calculate_probabilities=False  # Faster without probabilities
)

# Fit the model (only on documents with embeddings)
print("  Fitting BERTopic model...")
topics_valid, _ = topic_model.fit_transform(valid_docs, valid_embeddings)

# Create full topics array (-1 for documents without text)
topics = np.full(len(documents), -1, dtype=int)
topics[valid_indices] = topics_valid

# ── 5. Add Topics to Graph ────────────────────────────────────────────────
print("\n[5/6] Adding topics to graph nodes...")
for i, node in enumerate(node_ids):
    G.nodes[node]['topic'] = int(topics[i])

# Print topic statistics
unique_topics = np.unique(topics)
n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
outliers = np.sum(topics == -1)

print(f"  Number of topics found: {n_topics}")
print(f"  Outliers (topic -1): {outliers:,} ({100*outliers/len(topics):.1f}%)")

# Topic size distribution
topic_counts = pd.Series(topics).value_counts().sort_index()
print(f"\n  Topic size distribution:")
print(f"    Mean: {topic_counts[topic_counts.index != -1].mean():.1f}")
print(f"    Median: {topic_counts[topic_counts.index != -1].median():.1f}")
print(f"    Max: {topic_counts[topic_counts.index != -1].max()}")

# Show top topics
print(f"\n  Top 10 largest topics:")
for topic_id in topic_counts[topic_counts.index != -1].head(10).index:
    size = topic_counts[topic_id]
    words = topic_model.get_topic(int(topic_id))
    if words:
        top_words = ", ".join([word for word, _ in words[:5]])
        print(f"    Topic {topic_id:3d}: {size:5,} papers - {top_words}")

# ── 6. Save Updated Graph ─────────────────────────────────────────────────
print(f"\n[6/6] Saving updated graph to {OUTPUT_GRAPHML}...")
nx.write_graphml(G, OUTPUT_GRAPHML)
print(f"  Saved successfully!")

# ── 7. Save Topic Information ─────────────────────────────────────────────
print("\n[Bonus] Saving topic information...")

# Save topic info to CSV
topic_info = topic_model.get_topic_info()
topic_info.to_csv(DATA_DIR / "topic_info.csv", index=False)
print(f"  Saved topic_info.csv")

# Save document-topic mapping
doc_topics_df = pd.DataFrame({
    'node_id': node_ids,
    'topic': topics,
    'document': documents
})
doc_topics_df.to_csv(DATA_DIR / "document_topics.csv", index=False)
print(f"  Saved document_topics.csv")

# Save topic representations
topic_words = []
for topic_id in range(n_topics):
    words = topic_model.get_topic(topic_id)
    if words:
        word_list = [word for word, score in words]
        score_list = [score for word, score in words]
        topic_words.append({
            'topic_id': topic_id,
            'words': " | ".join(word_list),
            'scores': " | ".join([f"{s:.4f}" for s in score_list])
        })

topic_words_df = pd.DataFrame(topic_words)
topic_words_df.to_csv(DATA_DIR / "topic_words.csv", index=False)
print(f"  Saved topic_words.csv")

# Optional: Save the BERTopic model for future use
print("\n  Saving BERTopic model...")
topic_model.save(str(DATA_DIR / "bertopic_model"))
print(f"  Saved BERTopic model to {DATA_DIR / 'bertopic_model'}")

print("\n" + "=" * 80)
print("Topic modeling complete!")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Processed {len(documents):,} papers")
print(f"  - Found {n_topics} topics")
print(f"  - Updated graph saved to: {OUTPUT_GRAPHML}")
print(f"  - Topic info saved to: {DATA_DIR / 'topic_info.csv'}")
print(f"  - Document-topic mapping saved to: {DATA_DIR / 'document_topics.csv'}")
print(f"  - Topic words saved to: {DATA_DIR / 'topic_words.csv'}")