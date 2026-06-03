"""
Jaccard overlap between two nearest-neighbor dicts.

Conventions
-----------
- An NN dict has the form ``{node_id: [neighbor_ids]}`` with the source node
  excluded from its own list.
- Per-paper Jaccard counts only papers with full-length lists (len == k) on
  both sides; papers excluded are returned separately for accounting.
- The null baseline is a label permutation: it randomly reassigns one
  dict's NN lists to papers, preserving the marginal list-size
  distribution and the pool of neighbor lists, but breaking the link
  between a source paper and its own neighbors. It measures how far
  the observed Jaccard sits from chance pairing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_paper_jaccard(
    nn_a: dict[str, list[str]],
    nn_b: dict[str, list[str]],
    k: int,
    strict: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """Per-paper Jaccard between two NN dicts.

    Parameters
    ----------
    nn_a, nn_b
        neighbor-list dicts. Only papers in ``nn_a.keys() & nn_b.keys()``
        are considered.
    k
        Expected list length. Used by ``strict`` to decide inclusion.
    strict
        If True, only papers whose lists have length ``k`` on both sides
        contribute. Recommended for clean interpretation.

    Returns
    -------
    jaccards : dict[node_id, float]
        Per-paper Jaccards for included papers.
    excluded : list[node_id]
        Papers dropped because at least one list was shorter than ``k``
        (or because the union of lists was empty).
    """
    jaccards: dict[str, float] = {}
    excluded: list[str] = []
    for node in nn_a.keys() & nn_b.keys():
        a = set(nn_a[node])
        b = set(nn_b[node])
        if strict and (len(a) < k or len(b) < k):
            excluded.append(node)
            continue
        # defensive: in case a shuffled assignment puts the source in its own list
        a.discard(node)
        b.discard(node)
        union = a | b
        if not union:
            excluded.append(node)
            continue
        jaccards[node] = len(a & b) / len(union)
    return jaccards, excluded


def aggregate_jaccard(
    jaccards: dict[str, float],
    groups: dict[str, object] | None = None,
    group_name: str = "group",
) -> pd.DataFrame:
    """Mean, median, std and count of per-paper Jaccard, optionally by group.

    If ``groups`` is None, returns a single-row DataFrame for the overall
    aggregate. Otherwise, papers without a group assignment are dropped.
    """
    df = pd.DataFrame(
        {"node_id": list(jaccards.keys()), "jaccard": list(jaccards.values())}
    )
    if groups is None:
        df[group_name] = "_overall_"
    else:
        df[group_name] = df["node_id"].map(groups)
        df = df.dropna(subset=[group_name])

    agg = (
        df.groupby(group_name, dropna=False)["jaccard"]
        .agg(n="size", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def shuffle_null(
    nn_a: dict[str, list[str]],
    nn_b: dict[str, list[str]],
    k: int,
    n_runs: int = 100,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Mean / median Jaccard under random reassignment of ``nn_b`` lists.

    Each run permutes which paper receives which ``nn_b`` list (preserving
    the pool of lists and their marginal sizes, but breaking the paper→list
    pairing) and recomputes per-paper Jaccard under ``strict=True``. The
    resulting distribution is the chance baseline for the observed overlap.
    """
    rng = np.random.default_rng(seed)
    common = sorted(nn_a.keys() & nn_b.keys())
    nn_b_lists = [nn_b[n] for n in common]

    records = []
    for run in range(n_runs):
        perm = rng.permutation(len(common))
        nn_b_shuffled = {common[i]: nn_b_lists[perm[i]] for i in range(len(common))}
        jacc, _ = per_paper_jaccard(nn_a, nn_b_shuffled, k=k, strict=True)
        if jacc:
            vals = np.fromiter(jacc.values(), dtype=np.float64, count=len(jacc))
            records.append(
                {
                    "run": run,
                    "n": len(vals),
                    "mean": vals.mean(),
                    "median": float(np.median(vals)),
                }
            )
    return pd.DataFrame(records)
