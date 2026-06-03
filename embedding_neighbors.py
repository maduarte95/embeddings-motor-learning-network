"""
Embedding-space nearest neighbors.

Given a (n, d) array of paper embeddings and a parallel list of node ids,
return the top-k cosine-nearest neighbors for each paper.

Papers with zero-norm embeddings (e.g. no abstract → zero vector) are
excluded from *both* the query set (they receive an empty neighbor list)
and the candidate pool (they cannot appear as a neighbor).

Adding new metrics
------------------

If we later want a different distance (e.g. Euclidean), add another function `euclidean_nearest_neighbors`.
"""

from __future__ import annotations

import numpy as np


def cosine_nearest_neighbors(
    embeddings: np.ndarray,
    node_ids: list[str],
    k: int = 10,
) -> dict[str, list[str]]:
    """Top-k cosine-nearest neighbors per row.

    Parameters
    ----------
    embeddings
        Shape (n, d). Need not be L2-normalised; normalisation is done here.
    node_ids
        Length-n list mapping row index to node id.
    k
        Number of neighbors per paper.

    Returns
    -------
    dict[node_id, list[node_id]]
        neighbor ids ordered by descending cosine similarity. Papers with
        zero-norm embeddings return an empty list and are also excluded
        from every other paper's neighbor set.
    """
    if len(node_ids) != embeddings.shape[0]:
        raise ValueError(
            f"node_ids length ({len(node_ids)}) does not match "
            f"embeddings.shape[0] ({embeddings.shape[0]})."
        )

    X = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1)
    valid_mask = norms > 0
    valid_idx = np.where(valid_mask)[0]
    valid_ids = [node_ids[i] for i in valid_idx]

    Xv = X[valid_idx] / norms[valid_idx, None]
    sims = Xv @ Xv.T
    np.fill_diagonal(sims, -np.inf)

    n_valid = sims.shape[0]
    if n_valid == 0:
        return {nid: [] for nid in node_ids}

    k_eff = min(k, n_valid - 1)
    if k_eff <= 0:
        top_idx = np.empty((n_valid, 0), dtype=np.int64)
    else:
        part = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        row_idx = np.arange(n_valid)[:, None]
        top_scores = sims[row_idx, part]
        order_within = np.argsort(-top_scores, axis=1)
        top_idx = part[row_idx, order_within]

    out: dict[str, list[str]] = {}
    for i, vid in enumerate(valid_ids):
        out[vid] = [valid_ids[j] for j in top_idx[i]]
    for i, nid in enumerate(node_ids):
        if not valid_mask[i]:
            out[nid] = []
    return out
