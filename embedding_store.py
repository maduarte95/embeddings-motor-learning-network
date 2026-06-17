"""
Self-describing, per-model embedding storage.

Each embedding space is stored at ``data/embeddings/{key}.npz`` containing:

  - ``embeddings`` : float32 array, shape (n, d)
  - ``node_ids``   : unicode array, shape (n,) — row-aligned node ids and the
                     single source of truth for alignment
  - ``meta``       : JSON string {key, model, fingerprint, created, n, d}

Downstream code aligns BY node_id via :func:`align` (never by trusting row
order) and selects a space by ``key`` (e.g. "specter2", "scibert"). This
replaces the previous single hardcoded ``embeddings_cache.npz``, whose
alignment relied on matching row order with ``document_topics_new.csv`` and
which recorded no model identity.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

EMB_DIR = Path("data") / "embeddings"


def path_for(key: str) -> Path:
    """Path to the npz holding embedding space ``key``."""
    return EMB_DIR / f"{key}.npz"


def save_embeddings(
    key: str,
    embeddings: np.ndarray,
    node_ids: list[str],
    model: str,
    fingerprint: str | None = None,
) -> Path:
    """Write a self-describing embedding file for ``key``.

    ``embeddings[i]`` must correspond to ``node_ids[i]``.
    """
    if embeddings.shape[0] != len(node_ids):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != node_ids "
            f"({len(node_ids)}); refusing to save misaligned embeddings."
        )
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": key,
        "model": model,
        "fingerprint": fingerprint,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "n": int(embeddings.shape[0]),
        "d": int(embeddings.shape[1]),
    }
    out = path_for(key)
    np.savez_compressed(
        out,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        node_ids=np.asarray([str(n) for n in node_ids]),
        meta=json.dumps(meta),
    )
    return out


def load_embeddings(key: str) -> tuple[np.ndarray, list[str], dict]:
    """Load ``(embeddings, node_ids, meta)`` for embedding space ``key``."""
    p = path_for(key)
    if not p.exists():
        raise FileNotFoundError(
            f"No embeddings for key {key!r} at {p}. "
            "Run topic_modeling_new.py (or the relevant loader) first."
        )
    data = np.load(p)
    emb = data["embeddings"]
    node_ids = data["node_ids"].astype(str).tolist()
    meta = json.loads(str(data["meta"]))
    return emb, node_ids, meta


def align(
    embeddings: np.ndarray,
    node_ids: list[str],
    target_node_ids: list[str],
) -> np.ndarray:
    """Reorder ``embeddings`` rows to match ``target_node_ids``.

    Raises ``KeyError`` if any target id is missing from ``node_ids`` —
    replacing the old length-only check that could silently accept a
    row-order mismatch.
    """
    pos = {nid: i for i, nid in enumerate(node_ids)}
    missing = [t for t in target_node_ids if t not in pos]
    if missing:
        raise KeyError(
            f"{len(missing)} target node_ids absent from embeddings "
            f"(e.g. {missing[:3]})."
        )
    idx = np.fromiter(
        (pos[t] for t in target_node_ids),
        dtype=np.int64,
        count=len(target_node_ids),
    )
    return embeddings[idx]
