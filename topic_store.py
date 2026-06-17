"""
Per-embedding-model topic artifacts.

Topics are derived from a specific embedding space, so each model's BERTopic
outputs are stored under that model's key (the same key used by
``embedding_store``):

  - ``data/document_topics_{key}.csv``  node_id, topic, document
  - ``data/topic_words_{key}.csv``      topic_id, words, scores
  - ``data/topic_info_{key}.csv``       BERTopic get_topic_info() dump
  - ``data/bertopic_model_{key}/``      saved BERTopic model

This keeps a gemini run and a specter2 run from clobbering each other, and lets
a driver select a *consistent* (embeddings, topics) pair via one ``--embedding``
key. The citation graph (``citation_network_with_topics_new.graphml``) is shared
and model-independent for citations and Leiden ``cluster``; its ``topic`` node
attribute reflects only the most recent topic run, so for topic *analysis* the
keyed CSVs here are the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def doc_topics_path(key: str) -> Path:
    return DATA_DIR / f"document_topics_{key}.csv"


def topic_words_path(key: str) -> Path:
    return DATA_DIR / f"topic_words_{key}.csv"


def topic_info_path(key: str) -> Path:
    return DATA_DIR / f"topic_info_{key}.csv"


def model_dir(key: str) -> Path:
    return DATA_DIR / f"bertopic_model_{key}"


def load_doc_topics(key: str) -> pd.DataFrame:
    """Load the document-topic table for embedding ``key`` (node_id, topic, ...)."""
    p = doc_topics_path(key)
    if not p.exists():
        raise FileNotFoundError(
            f"No topics for key {key!r} at {p}. "
            f"Run: pixi run python topic_modeling_new.py --embedding {key}"
        )
    return pd.read_csv(p)


def load_topic_words(key: str) -> pd.DataFrame:
    """Load the topic-words table for embedding ``key`` (topic_id, words, scores)."""
    p = topic_words_path(key)
    if not p.exists():
        raise FileNotFoundError(
            f"No topic words for key {key!r} at {p}. "
            f"Run: pixi run python topic_modeling_new.py --embedding {key}"
        )
    return pd.read_csv(p)
