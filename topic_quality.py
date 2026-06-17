"""
Topic-quality metrics — the standard yardstick for "are the topics good" and,
held under a fixed pipeline, for comparing embedding models (the swap test).

All functions are pure and composable; nothing here re-runs BERTopic. They take
a topic *labelling* (+ the corpus / top words / authors) so the same code scores
any model's topics, loaded by key from topic_store.

Metrics
-------
- ``npmi_coherence``  : do each topic's top words actually co-occur in the
  abstracts? Normalised PMI (Bouma 2009; Lau et al. 2014), computed directly
  from document co-occurrence with sklearn's CountVectorizer so the reference
  tokenisation (incl. bigrams) matches BERTopic's vocabulary. No gensim needed.
- ``topic_diversity``: fraction of unique words across topics' top-n (Dieng
  et al. 2020) — catches topics that all repeat the same generic words.
- ``outlier_fraction``: share of documents in the -1 bucket (coverage guard;
  see [[embedding-evaluation-plan]] — coverage rises as you coarsen, so it must
  be read alongside coherence, never maximised on its own).
- ``author_overlap_within_topics``: DIAGNOSTIC (not a score to maximise) — mean
  number of shared authors per within-topic paper pair vs a label-shuffle null.
  Tells you how much topics coincide with author communities. Exact-name match,
  so it under-counts spelling variants.

For coherence/diversity, *higher is better*. For author overlap there is no
"better" — a good topic could be a tight lab community OR a broad cross-lab
subfield; the z-score just says whether grouping concentrates co-authorship.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# ── Coverage ─────────────────────────────────────────────────────────────────

def outlier_fraction(labels) -> float:
    """Fraction of documents assigned to the -1 (outlier) bucket."""
    labels = np.asarray(labels)
    return float((labels == -1).mean()) if labels.size else float("nan")


# ── Diversity ────────────────────────────────────────────────────────────────

def topic_diversity(topic_word_lists: list[list[str]], top_n: int = 10) -> float:
    """Fraction of unique words across all topics' top-n words (Dieng 2020).

    1.0 = every topic word is unique; low = topics share generic vocabulary.
    """
    seen: set[str] = set()
    total = 0
    for words in topic_word_lists:
        for w in words[:top_n]:
            seen.add(w)
            total += 1
    return len(seen) / total if total else float("nan")


# ── Coherence (NPMI) ─────────────────────────────────────────────────────────

def _npmi_for_topic(idx: list[int], C: np.ndarray, df: np.ndarray, N: int) -> float:
    """Mean pairwise NPMI over the (valid) word indices of one topic."""
    valid = [i for i in idx if df[i] > 0]
    if len(valid) < 2:
        return float("nan")
    vals = []
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            i, j = valid[a], valid[b]
            c_ij = C[i, j]
            if c_ij == 0:
                vals.append(-1.0)  # never co-occur -> NPMI = -1 by convention
                continue
            p_ij = c_ij / N
            p_i = df[i] / N
            p_j = df[j] / N
            vals.append((math.log(p_ij) - math.log(p_i) - math.log(p_j)) / (-math.log(p_ij)))
    return sum(vals) / len(vals) if vals else float("nan")


def npmi_coherence(
    topic_word_lists: list[list[str]],
    docs: list[str],
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> tuple[float, dict]:
    """Mean topic NPMI coherence and a per-topic dict.

    Parameters
    ----------
    topic_word_lists
        One list of top words per topic (order = score order).
    docs
        Reference corpus (the documents the topics were built on).
    top_n, ngram_range, stop_words
        ``ngram_range`` / ``stop_words`` should match the BERTopic vectoriser so
        bigram topic words can be found in the reference corpus.

    Returns
    -------
    (mean_coherence, {topic_position: coherence})
        ``topic_position`` indexes ``topic_word_lists``.
    """
    topics = [list(words[:top_n]) for words in topic_word_lists]
    vocab = sorted({w for t in topics for w in t})
    if not vocab:
        return float("nan"), {}

    cv = CountVectorizer(
        vocabulary=vocab, ngram_range=ngram_range,
        stop_words=stop_words, binary=True, lowercase=True,
    )
    X = cv.transform(docs)                       # (n_docs, |vocab|) binary
    N = X.shape[0]
    df = np.asarray(X.sum(axis=0)).ravel()       # document frequency per term
    C = (X.T @ X).toarray()                       # co-document counts (|V| x |V|)
    col = cv.vocabulary_                          # term -> column index

    per_topic = {}
    for pos, words in enumerate(topics):
        idx = [col[w] for w in words if w in col]
        per_topic[pos] = _npmi_for_topic(idx, C, df, N)
    vals = [v for v in per_topic.values() if not math.isnan(v)]
    mean = sum(vals) / len(vals) if vals else float("nan")
    return mean, per_topic


# ── Author overlap within topics (diagnostic) ────────────────────────────────

def parse_authors(value) -> set[str]:
    """Parse a pipe-delimited authors string into a set (exact-name match)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {a.strip() for a in str(value).split("|") if a.strip()}


def _author_pair_rate(labels: np.ndarray, author_sets: list[set[str]],
                      exclude_outlier: bool) -> tuple[float, dict]:
    """Mean shared-authors per within-topic paper pair, overall and per topic.

    Per topic: sum_a C(k_a, 2) / C(n, 2), where k_a = #papers in the topic by
    author a — i.e. the mean number of shared authors over all paper pairs.
    """
    counts: dict = defaultdict(lambda: defaultdict(int))
    sizes: dict = defaultdict(int)
    for t, authors in zip(labels, author_sets):
        if exclude_outlier and t == -1:
            continue
        sizes[t] += 1
        for a in authors:
            counts[t][a] += 1
    total_shared = total_pairs = 0.0
    per_topic = {}
    for t, n in sizes.items():
        pairs = n * (n - 1) / 2
        shared = sum(k * (k - 1) / 2 for k in counts[t].values())
        per_topic[t] = shared / pairs if pairs > 0 else float("nan")
        total_shared += shared
        total_pairs += pairs
    overall = total_shared / total_pairs if total_pairs > 0 else float("nan")
    return overall, per_topic


def author_overlap_within_topics(
    topic_by_node: dict,
    authors_by_node: dict,
    n_null: int = 100,
    seed: int = 0,
    exclude_outlier: bool = True,
) -> dict:
    """Within-topic author cohesion vs a label-shuffle null.

    Returns ``{observed, null_mean, null_std, z, per_topic}``. A large positive
    ``z`` means topics concentrate co-authorship far more than chance. This is a
    descriptor, not a target — see module docstring.
    """
    nodes = [n for n in topic_by_node if n in authors_by_node]
    labels = np.array([topic_by_node[n] for n in nodes])
    author_sets = [authors_by_node[n] for n in nodes]

    observed, per_topic = _author_pair_rate(labels, author_sets, exclude_outlier)

    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    null = np.empty(n_null)
    for i in range(n_null):
        rng.shuffle(shuffled)
        null[i] = _author_pair_rate(shuffled, author_sets, exclude_outlier)[0]

    std = float(null.std())
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_std": std,
        "z": (observed - null.mean()) / std if std > 0 else float("nan"),
        "per_topic": per_topic,
    }


# ── Convenience: score a model's saved topics ────────────────────────────────

def _word_lists_from_df(topic_words_df: pd.DataFrame, top_n: int) -> list[list[str]]:
    out = []
    for _, row in topic_words_df.iterrows():
        words = [w.strip() for w in str(row["words"]).split("|")]
        out.append([w for w in words if w][:top_n])
    return out


def evaluate_topics(
    key: str,
    authors_by_node: dict | None = None,
    top_n: int = 10,
    n_null: int = 100,
) -> dict:
    """Load a model's saved topics and return all metrics in one dict.

    ``authors_by_node`` is optional (read from the graph by callers); when
    omitted the author diagnostic is skipped.
    """
    from topic_store import load_doc_topics, load_topic_words

    doc = load_doc_topics(key)
    tw = load_topic_words(key)
    word_lists = _word_lists_from_df(tw, top_n)

    mean_npmi, per_topic_npmi = npmi_coherence(
        word_lists, doc["document"].fillna("").tolist(), top_n=top_n
    )
    result = {
        "key": key,
        "n_topics": int((doc["topic"] >= 0).any() and doc.loc[doc["topic"] >= 0, "topic"].nunique()),
        "npmi": mean_npmi,
        "diversity": topic_diversity(word_lists, top_n=top_n),
        "outlier_pct": 100 * outlier_fraction(doc["topic"].to_numpy()),
    }
    if authors_by_node is not None:
        topic_by_node = {str(n): int(t) for n, t in zip(doc["node_id"], doc["topic"])}
        ao = author_overlap_within_topics(topic_by_node, authors_by_node, n_null=n_null)
        result["author_overlap"] = ao["observed"]
        result["author_overlap_z"] = ao["z"]
    return result


if __name__ == "__main__":
    import argparse

    import networkx as nx

    from embedding_loaders import DEFAULT_EMBEDDING

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                        help=f"Embedding model key (default: {DEFAULT_EMBEDDING}).")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--graphml", default="data/citation_network_with_topics_new.graphml",
                        help="Graph providing the 'authors' attribute.")
    args = parser.parse_args()

    print(f"Loading authors from {args.graphml} ...")
    G = nx.read_graphml(args.graphml)
    authors_by_node = {n: parse_authors(d.get("authors")) for n, d in G.nodes(data=True)}

    res = evaluate_topics(
        args.embedding, authors_by_node=authors_by_node,
        top_n=args.top_n, n_null=args.n_null,
    )
    print(f"\nTopic quality — model: {res['key']}")
    print(f"  topics (excl. -1)   : {res['n_topics']}")
    print(f"  NPMI coherence      : {res['npmi']:.4f}   (higher = better)")
    print(f"  topic diversity     : {res['diversity']:.4f}   (higher = better)")
    print(f"  outliers (topic -1) : {res['outlier_pct']:.1f}%")
    print(f"  author overlap      : {res['author_overlap']:.4f}  "
          f"(z={res['author_overlap_z']:.1f} vs label-shuffle null; diagnostic)")
