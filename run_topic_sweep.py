"""
Topic-granularity sweep: pick a defensible BERTopic ``min_cluster_size``.

Holds everything upstream fixed (embeddings, and the 5-D clustering UMAP computed
once) and varies only HDBSCAN's ``min_cluster_size``, scoring each setting with
topic_quality. The topic count is emergent — you read it off, you don't set it.

For each min_cluster_size it tabulates:
  - n_topics        : distinct topics (excl. -1)
  - coverage        : 1 - outlier fraction
  - npmi            : mean NPMI coherence   (higher = better)
  - diversity       : unique-word fraction  (higher = better)

Read it as a trade-off, not a single maximum: coverage rises monotonically as you
coarsen (don't chase it), while coherence/diversity usually peak in the middle.
See [[embedding-evaluation-plan]].

Note: this re-runs HDBSCAN (and c-TF-IDF) from scratch for every grid point — it
does NOT use the cached BERTopic model, and the corpus fingerprint does not bust
that cache on a parameter change, which is exactly why the sweep drives HDBSCAN
directly instead of calling topic_modeling_new.py.

Output: data/topic_sweep_{key}.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from embedding_loaders import DEFAULT_EMBEDDING
from embedding_store import load_embeddings
from topic_quality import npmi_coherence, topic_diversity, outlier_fraction

# Clustering params — kept identical to topic_modeling_new.py so the sweep is
# representative of the real pipeline (only min_cluster_size varies).
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_SEED = 42
VECTORIZER_NGRAM = (1, 2)
VECTORIZER_MAX_FEATURES = 10000
TOP_N_WORDS = 10

DATA_DIR = Path("data")
DEFAULT_GRID = [10, 15, 20, 30, 50, 75, 100]


class _IdentityReduction:
    """Pass-through 'UMAP' so BERTopic clusters on precomputed coords."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def _fingerprint(emb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{emb.shape}|{UMAP_N_NEIGHBORS}|{UMAP_N_COMPONENTS}|"
             f"{UMAP_MIN_DIST}|{UMAP_METRIC}|{UMAP_SEED}".encode())
    h.update(np.ascontiguousarray(emb, dtype=np.float32).tobytes())
    return h.hexdigest()


def reduced_layout(emb: np.ndarray, key: str, recompute: bool = False) -> np.ndarray:
    """5-D clustering UMAP, cached per embedding key at data/umap_5d_{key}.npz."""
    cache = DATA_DIR / f"umap_5d_{key}.npz"
    fp = _fingerprint(emb)
    if not recompute and cache.exists():
        cached = np.load(cache)
        if str(cached["fingerprint"]) == fp:
            print(f"  5-D UMAP cache hit — {cache.name}")
            return cached["coords"]

    print(f"  Computing 5-D UMAP on {emb.shape} (once)...")
    from umap import UMAP

    coords = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=UMAP_SEED,
    ).fit_transform(emb).astype(np.float32)
    np.savez_compressed(cache, coords=coords, fingerprint=fp)
    print(f"  5-D UMAP saved to {cache.name}")
    return coords


def docs_for(node_ids: list[str], graphml: Path) -> list[str]:
    """Reconstruct 'title [SEP] abstract' per node, matching topic_modeling_new."""
    G = nx.read_graphml(graphml)
    out = []
    for nid in node_ids:
        d = G.nodes[nid]
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        out.append(f"{title} [SEP] {abstract}" if abstract else title)
    return out


def cluster_at(reduced: np.ndarray, docs: list[str], min_cluster_size: int):
    """Run BERTopic (identity reducer + HDBSCAN) and return (labels, word_lists)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    model = BERTopic(
        umap_model=_IdentityReduction(),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size, metric="euclidean",
            cluster_selection_method="eom", prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english", max_features=VECTORIZER_MAX_FEATURES,
            ngram_range=VECTORIZER_NGRAM,
        ),
        top_n_words=TOP_N_WORDS, calculate_probabilities=False, verbose=False,
    )
    labels = np.array(model.fit_transform(docs, reduced)[0])
    topic_ids = sorted(t for t in set(labels) if t != -1)
    word_lists = [[w for w, _ in (model.get_topic(t) or [])] for t in topic_ids]
    return labels, word_lists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                        help=f"Embedding model key (default: {DEFAULT_EMBEDDING}).")
    parser.add_argument("--min-cluster-sizes", default=",".join(map(str, DEFAULT_GRID)),
                        help="Comma-separated grid (default: %(default)s).")
    parser.add_argument("--graphml", default=str(DATA_DIR / "citation_network_with_topics_new.graphml"))
    parser.add_argument("--recompute-umap", action="store_true",
                        help="Ignore the cached 5-D clustering UMAP.")
    args = parser.parse_args()

    grid = [int(x) for x in args.min_cluster_sizes.split(",") if x.strip()]

    print(f"Loading embeddings (model: {args.embedding})...")
    emb, node_ids, meta = load_embeddings(args.embedding)
    print(f"  {emb.shape}  model={meta.get('model')}")

    reduced = reduced_layout(emb, args.embedding, recompute=args.recompute_umap)
    print("Reconstructing documents...")
    docs = docs_for(node_ids, Path(args.graphml))

    rows = []
    for m in grid:
        print(f"\n=== min_cluster_size = {m} ===")
        labels, word_lists = cluster_at(reduced, docs, m)
        n_topics = len(word_lists)
        cov = 1.0 - outlier_fraction(labels)
        npmi = npmi_coherence(word_lists, docs, top_n=TOP_N_WORDS)[0]
        div = topic_diversity(word_lists, top_n=TOP_N_WORDS)
        print(f"  n_topics={n_topics}  coverage={cov:.3f}  npmi={npmi:.4f}  diversity={div:.4f}")
        rows.append({
            "min_cluster_size": m, "n_topics": n_topics,
            "coverage": cov, "npmi": npmi, "diversity": div,
        })

    out = pd.DataFrame(rows)
    out_path = DATA_DIR / f"topic_sweep_{args.embedding}.csv"
    out.to_csv(out_path, index=False)
    print("\n=== Sweep summary ===")
    print(out.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
