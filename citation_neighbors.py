"""
Citation-space nearest neighbors.

Each "similarity method" maps a paper to its top-k most related papers via
some citation-graph-derived proximity. Currently supported:

  - bibliographic_coupling : Salton cosine of out-citation (reference) vectors.
    Reference: Kessler (1963), Am. Documentation 14(1), 10-25.
  - co_citation            : Salton cosine of in-citation (cited-by) vectors.
    Reference: Small (1973), JASIS 24(4), 265-269.
  - combined               : 0.5 * (bibliographic_coupling + co_citation).
    Pragmatic synthesis; no canonical single reference (cf. Boyack & Klavans
    2010, JASIST 61(12), for hybrid citation mapping at the network level).

Adding new methods
------------------

For a similarity that produces an n x n matrix (e.g. personalised PageRank,
Adamic-Adar, Katz):

    def my_measure(A: sp.csr_matrix) -> sp.csr_matrix:
        # ... return similarity with zero diagonal
    SIMILARITY_FUNCTIONS["my_measure"] = my_measure

For a method whose natural output is a neighbor list (e.g. direct edges,
shortest-path k-NN):

    def my_neighbors(G: nx.DiGraph, k: int) -> dict[str, list[str]]:
        # ...
    NEIGHBOR_FUNCTIONS["my_neighbors"] = my_neighbors

The dispatcher `nearest_neighbors(G, method, k)` resolves either kind.
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp


# --- Adjacency -----------------------------------------------------------

def build_adjacency(G: nx.DiGraph) -> tuple[sp.csr_matrix, list[str]]:
    """Sparse 0/1 adjacency with A[i, j] = 1 iff node i cites node j.

    Returns the matrix and the row-to-node-id mapping.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if G.number_of_edges() == 0:
        return sp.csr_matrix((n, n), dtype=np.float32), nodes

    idx = {node: i for i, node in enumerate(nodes)}
    m = G.number_of_edges()
    rows = np.fromiter((idx[u] for u, _ in G.edges()), dtype=np.int64, count=m)
    cols = np.fromiter((idx[v] for _, v in G.edges()), dtype=np.int64, count=m)
    data = np.ones(m, dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), nodes


# --- Helpers -------------------------------------------------------------

def _salton_normalize(S: sp.csr_matrix, norms_sq: np.ndarray) -> sp.csr_matrix:
    """Divide S[i, j] by sqrt(norms_sq[i] * norms_sq[j]); zero where the norm is zero."""
    S = S.tocoo()
    denom = np.sqrt(
        norms_sq[S.row].astype(np.float64) * norms_sq[S.col].astype(np.float64)
    )
    new_data = np.zeros_like(S.data, dtype=np.float32)
    nz = denom > 0
    new_data[nz] = (S.data[nz] / denom[nz]).astype(np.float32)
    return sp.csr_matrix((new_data, (S.row, S.col)), shape=S.shape)


def _zero_diagonal(S: sp.csr_matrix) -> sp.csr_matrix:
    S = S.tolil()
    S.setdiag(0)
    return S.tocsr()


# --- Similarity measures (matrix-producing) ------------------------------

def bibliographic_coupling(A: sp.csr_matrix) -> sp.csr_matrix:
    """B[i, j] = |refs(i) ∩ refs(j)| / sqrt(|refs(i)| * |refs(j)|)."""
    B = A @ A.T
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    return _zero_diagonal(_salton_normalize(B, out_deg))


def co_citation(A: sp.csr_matrix) -> sp.csr_matrix:
    """C[i, j] = |citers(i) ∩ citers(j)| / sqrt(|citers(i)| * |citers(j)|)."""
    C = A.T @ A
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    return _zero_diagonal(_salton_normalize(C, in_deg))


def combined(A: sp.csr_matrix) -> sp.csr_matrix:
    """Mean of bibliographic_coupling and co_citation."""
    return (bibliographic_coupling(A) + co_citation(A)) * 0.5


# --- Top-k extraction ----------------------------------------------------

def top_k_from_sparse(
    S: sp.csr_matrix,
    k: int,
    node_ids: list[str],
) -> dict[str, list[str]]:
    """For each row, return up to k column indices with the largest values.

    Rows with fewer than k non-zero entries return however many neighbors
    exist; rows with no non-zero entries return an empty list.
    """
    S = S.tocsr()
    n = S.shape[0]
    out: dict[str, list[str]] = {}
    for i in range(n):
        start, end = S.indptr[i], S.indptr[i + 1]
        cols = S.indices[start:end]
        vals = S.data[start:end]
        if len(vals) == 0:
            out[node_ids[i]] = []
            continue
        if len(vals) <= k:
            order = np.argsort(-vals)
        else:
            part = np.argpartition(-vals, k - 1)[:k]
            order = part[np.argsort(-vals[part])]
        out[node_ids[i]] = [node_ids[c] for c in cols[order]]
    return out


# --- Dispatcher ----------------------------------------------------------

SIMILARITY_FUNCTIONS: dict[str, Callable[[sp.csr_matrix], sp.csr_matrix]] = {
    "bibliographic_coupling": bibliographic_coupling,
    "co_citation": co_citation,
    "combined": combined,
}

NEIGHBOR_FUNCTIONS: dict[str, Callable[..., dict[str, list[str]]]] = {}


def nearest_neighbors(
    G: nx.DiGraph,
    method: str = "bibliographic_coupling",
    k: int = 10,
) -> dict[str, list[str]]:
    """Top-k citation-space neighbors per node under the chosen similarity.

    Parameters
    ----------
    G
        Directed citation graph. Edge (u, v) means u cites v.
    method
        Name registered in SIMILARITY_FUNCTIONS or NEIGHBOR_FUNCTIONS.
    k
        Number of neighbors per node.
    """
    if method in SIMILARITY_FUNCTIONS:
        A, nodes = build_adjacency(G)
        S = SIMILARITY_FUNCTIONS[method](A)
        return top_k_from_sparse(S, k, nodes)
    if method in NEIGHBOR_FUNCTIONS:
        return NEIGHBOR_FUNCTIONS[method](G, k=k)
    available = sorted(set(SIMILARITY_FUNCTIONS) | set(NEIGHBOR_FUNCTIONS))
    raise ValueError(f"Unknown method {method!r}. Available: {available}")
