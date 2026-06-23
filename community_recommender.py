"""
Recommender: within a semantic topic, surface citation-community pairs that are
*semantically together but far apart in citation space*.

Design
------
The BERTopic topic is taken as the **semantic ground truth**: two citation
communities that both have papers in a topic are, by construction, working on
related material ("they should be talking"). The only thing measured in
citation space is how *connected* each pair is ("are they talking"). Bottom-k
least-connected pairs within a topic are the recommendations: same problem
space, but not linked through citations.

Connectedness (citation space) — pluggable ``method``
-----------------------------------------------------
Default is **bibliographic coupling** at the community-pair level. Each
community gets a reference-count profile ``M = S @ A`` (S = community->paper
membership, A = paper-cites-paper adjacency); coupling(A, B) is the cosine of
the two communities' reference profiles — i.e. *how much they cite the same
literature*. Low coupling -> they draw on different literatures despite being
in the same topic -> recommended pair.

Because coupling is a cosine, it is already symmetrically size-normalised
(both magnitudes sit in the denominator), so it does **not** suffer the
partner-size baseline problem that raw conductance does — no null/ratio
correction is needed. (Residual caveat: "hub" references everyone cites can
inflate coupling; not corrected here.)

Available methods (all reduce to a community x community score matrix):
  - ``bibliographic_coupling`` : cosine of out-citation (reference) profiles.
  - ``co_citation``            : cosine of in-citation (cited-by) profiles.
  - ``combined``               : mean of the two.
  - ``conductance``            : first-order direct-edge cut / min(volume).
    Kept as an option; note its size-baseline caveat (ranks low for big,
    prolific communities regardless of whether they actually avoid each other).

For every method, **lower score = less connected**, so ranking is ascending.

Entry points
------------
- ``recommend_for_topic(topic, ...)``  the core operation.
- ``recommend_for_paper(node_id, ...)`` -> the paper's topic -> topic recs.
- ``recommend_for_author(query, ...)``  -> topics the author appears in.

Outputs go to ``data/recommender_results/``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from citation_neighbors import build_adjacency
from embedding_loaders import DEFAULT_EMBEDDING
from topic_store import load_doc_topics

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "recommender_results"
GRAPHML = DATA_DIR / "citation_network_with_topics_new.graphml"

COMMUNITY_ATTR = "cluster"
OUTLIER_TOPIC = -1          # BERTopic noise bucket; never a real topic
DEFAULT_MIN_PAPERS = 10     # min papers a community needs *in a topic* to qualify
DEFAULT_K = 10
DEFAULT_METHOD = "bibliographic_coupling"
METHODS = ("bibliographic_coupling", "co_citation", "combined", "conductance")


# ---------------------------------------------------------------------------
# Community-pair connectedness (topic-independent: pure citation structure)
# ---------------------------------------------------------------------------

@dataclass
class Connectedness:
    """A community x community connectedness score matrix (lower = less connected)."""

    comm_ids: list[str]          # ordered community ids
    index: dict[str, int]        # community id -> row/col index
    score: np.ndarray            # score[i, j] = connectedness of comm i & j
    size: np.ndarray             # # nodes per community
    method: str

    def of(self, a: str, b: str) -> float:
        return float(self.score[self.index[a], self.index[b]])


def _membership_selector(
    G: nx.DiGraph, nodes: list[str], community_attr: str
) -> tuple[sp.csr_matrix, list[str], dict[str, int], np.ndarray]:
    """Sparse (nComm x nPaper) 0/1 matrix S[g, i] = 1 iff paper i is in community g.

    ``nodes`` is the paper ordering used by the adjacency matrix, so S lines up
    with A column/row-for-row.
    """
    comm_by_node = {
        n: str(a[community_attr])
        for n, a in G.nodes(data=True)
        if a.get(community_attr) is not None
    }
    comm_ids = sorted(set(comm_by_node.values()), key=_comm_sort_key)
    index = {c: i for i, c in enumerate(comm_ids)}

    rows, cols, size = [], [], np.zeros(len(comm_ids), dtype=np.int64)
    for col, node in enumerate(nodes):
        c = comm_by_node.get(node)
        if c is None:
            continue
        g = index[c]
        rows.append(g)
        cols.append(col)
        size[g] += 1
    S = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(comm_ids), len(nodes)),
    )
    return S, comm_ids, index, size


def _cosine_gram(M: sp.csr_matrix) -> np.ndarray:
    """Row-wise cosine similarity matrix of sparse M (zeros for zero-norm rows)."""
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    Mn = sp.diags(inv) @ M
    return np.asarray((Mn @ Mn.T).todense())


def compute_connectedness(
    G: nx.DiGraph,
    method: str = DEFAULT_METHOD,
    community_attr: str = COMMUNITY_ATTR,
) -> Connectedness:
    """Build the community x community connectedness matrix for ``method``."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available: {METHODS}")

    A, nodes = build_adjacency(G)  # A[i, j] = 1 iff paper i cites paper j
    S, comm_ids, index, size = _membership_selector(G, nodes, community_attr)

    if method == "conductance":
        score = _conductance_matrix(S, A)
    else:
        # Out-citation (reference) profile and in-citation (cited-by) profile.
        coupling = _cosine_gram(S @ A) if method != "co_citation" else None
        cocit = _cosine_gram(S @ A.T) if method != "bibliographic_coupling" else None
        if method == "bibliographic_coupling":
            score = coupling
        elif method == "co_citation":
            score = cocit
        else:  # combined
            score = 0.5 * (coupling + cocit)

    np.fill_diagonal(score, np.nan)  # self-pairs are meaningless
    return Connectedness(comm_ids, index, score, size, method)


def _conductance_matrix(S: sp.csr_matrix, A: sp.csr_matrix) -> np.ndarray:
    """phi(i, j) = (e(i->j) + e(j->i)) / min(vol(i), vol(j)); first-order edges."""
    E = np.asarray((S @ A @ S.T).todense())          # E[i, j] = edges comm i -> comm j
    outvol = np.asarray(A.sum(axis=1)).ravel()       # per-paper out-degree
    invol = np.asarray(A.sum(axis=0)).ravel()         # per-paper in-degree
    vol = np.asarray(S @ (outvol + invol)).ravel()   # per-community total degree
    cut = E + E.T
    denom = np.minimum.outer(vol, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, cut / denom, np.nan)
    return score


def _comm_sort_key(c: str):
    """Sort numeric community ids numerically, fall back to string."""
    try:
        return (0, int(c))
    except (TypeError, ValueError):
        return (1, c)


# ---------------------------------------------------------------------------
# Topic participation (semantic ground truth, from the keyed BERTopic run)
# ---------------------------------------------------------------------------

@dataclass
class TopicMembership:
    participation: dict[int, Counter]   # topic -> {community: # papers in topic}
    topic_by_node: dict[str, int]
    comm_by_node: dict[str, str]


def load_membership(
    G: nx.DiGraph, key: str, community_attr: str = COMMUNITY_ATTR
) -> TopicMembership:
    """Map each topic to the communities represented in it and by how many papers."""
    comm_by_node = {
        n: str(a[community_attr])
        for n, a in G.nodes(data=True)
        if a.get(community_attr) is not None
    }
    df = load_doc_topics(key)
    topic_by_node = dict(zip(df["node_id"].astype(str), df["topic"].astype(int)))

    participation: dict[int, Counter] = defaultdict(Counter)
    for n, c in comm_by_node.items():
        t = topic_by_node.get(n)
        if t is None or t == OUTLIER_TOPIC:
            continue
        participation[t][c] += 1

    return TopicMembership(dict(participation), topic_by_node, comm_by_node)


# ---------------------------------------------------------------------------
# Core recommendation
# ---------------------------------------------------------------------------

def recommend_for_topic(
    topic: int,
    conn: Connectedness,
    membership: TopicMembership,
    k: int | None = DEFAULT_K,
    min_papers: int = DEFAULT_MIN_PAPERS,
) -> pd.DataFrame:
    """Bottom-k least-connected community pairs among a topic's communities.

    Returns a DataFrame sorted ascending by ``connectedness`` (most disconnected
    first). ``k=None`` returns every candidate pair (useful for inspection).
    """
    part = membership.participation.get(topic)
    if not part:
        return _empty_recs()

    candidates = sorted(
        (c for c, n in part.items() if n >= min_papers), key=_comm_sort_key
    )
    if len(candidates) < 2:
        return _empty_recs()

    rows = []
    for a, b in combinations(candidates, 2):
        i, j = conn.index[a], conn.index[b]
        rows.append({
            "topic": topic,
            "community_a": a,
            "community_b": b,
            "n_a_in_topic": part[a],
            "n_b_in_topic": part[b],
            "size_a": int(conn.size[i]),
            "size_b": int(conn.size[j]),
            "connectedness": conn.score[i, j],
            "method": conn.method,
        })

    df = pd.DataFrame(rows).sort_values(
        "connectedness", ascending=True, kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    return df.head(k) if k is not None else df


def _empty_recs() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "topic", "community_a", "community_b", "n_a_in_topic", "n_b_in_topic",
        "size_a", "size_b", "connectedness", "method",
    ])


# ---------------------------------------------------------------------------
# Paper / author entry points (thin wrappers over recommend_for_topic)
# ---------------------------------------------------------------------------

def recommend_for_paper(
    node_id: str,
    conn: Connectedness,
    membership: TopicMembership,
    k: int | None = DEFAULT_K,
    min_papers: int = DEFAULT_MIN_PAPERS,
) -> pd.DataFrame:
    """Recommendations for the topic the given paper belongs to."""
    topic = membership.topic_by_node.get(str(node_id))
    if topic is None or topic == OUTLIER_TOPIC:
        return _empty_recs()
    df = recommend_for_topic(topic, conn, membership, k=k, min_papers=min_papers)
    return df.assign(via_node=str(node_id))


def recommend_for_author(
    query: str,
    G: nx.DiGraph,
    conn: Connectedness,
    membership: TopicMembership,
    k: int | None = DEFAULT_K,
    min_papers: int = DEFAULT_MIN_PAPERS,
) -> pd.DataFrame:
    """Recommendations across every (non-outlier) topic the author appears in.

    ``query`` is matched case-insensitively as a substring of the pipe-separated
    ``authors`` node attribute, so "rizzolatti" finds "Rizzolatti, G".
    """
    q = query.casefold()
    topics_for_author: set[int] = set()
    for n, a in G.nodes(data=True):
        authors = a.get("authors")
        if not authors or q not in str(authors).casefold():
            continue
        t = membership.topic_by_node.get(str(n))
        if t is not None and t != OUTLIER_TOPIC:
            topics_for_author.add(t)

    if not topics_for_author:
        return _empty_recs()

    parts = [
        recommend_for_topic(t, conn, membership, k=k, min_papers=min_papers)
        for t in sorted(topics_for_author)
    ]
    out = pd.concat(parts, ignore_index=True) if parts else _empty_recs()
    return out.assign(via_author=query)


def recommend_all_topics(
    conn: Connectedness,
    membership: TopicMembership,
    k: int | None = DEFAULT_K,
    min_papers: int = DEFAULT_MIN_PAPERS,
) -> pd.DataFrame:
    """Run every topic and stack the bottom-k tables (for a full sweep / export)."""
    frames = [
        recommend_for_topic(t, conn, membership, k=k, min_papers=min_papers)
        for t in sorted(membership.participation)
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _empty_recs()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                        help=f"Embedding/topic model key (default: {DEFAULT_EMBEDDING}).")
    parser.add_argument("--method", default=DEFAULT_METHOD, choices=METHODS,
                        help=f"Connectedness measure (default: {DEFAULT_METHOD}).")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--topic", type=int, help="Recommend for a single topic id.")
    target.add_argument("--paper", help="Recommend for a paper's topic (node id, e.g. n123).")
    target.add_argument("--author", help="Recommend across an author's topics (name substring).")
    target.add_argument("--all", action="store_true", help="Sweep every topic and export.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Bottom-k pairs to return (default: {DEFAULT_K}; 0 = all).")
    parser.add_argument("--min-papers", type=int, default=DEFAULT_MIN_PAPERS,
                        help=f"Min papers a community needs in a topic (default: {DEFAULT_MIN_PAPERS}).")
    args = parser.parse_args()
    k = None if args.k == 0 else args.k

    print(f"Loading graph from {GRAPHML}...")
    G = nx.read_graphml(GRAPHML)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, directed={G.is_directed()}")

    print(f"Computing community connectedness (method: {args.method})...")
    conn = compute_connectedness(G, method=args.method)
    print(f"  {len(conn.comm_ids)} communities")

    print(f"Loading topic membership (model: {args.embedding})...")
    membership = load_membership(G, args.embedding)
    print(f"  {len(membership.participation)} non-outlier topics")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        df = recommend_all_topics(conn, membership, k=k, min_papers=args.min_papers)
        out = OUT_DIR / f"recommendations_{args.embedding}_{args.method}.csv"
        df.to_csv(out, index=False)
        n_topics = df["topic"].nunique() if not df.empty else 0
        print(f"\n{len(df)} pairs across {n_topics} topics -> {out}")
        return

    if args.topic is not None:
        df = recommend_for_topic(args.topic, conn, membership, k=k, min_papers=args.min_papers)
        label = f"topic {args.topic}"
    elif args.paper:
        df = recommend_for_paper(args.paper, conn, membership, k=k, min_papers=args.min_papers)
        t = membership.topic_by_node.get(str(args.paper))
        label = f"paper {args.paper} (topic {t})"
    elif args.author:
        df = recommend_for_author(args.author, G, conn, membership, k=k, min_papers=args.min_papers)
        label = f"author '{args.author}'"
    else:
        parser.error("choose one of --topic / --paper / --author / --all")

    print(f"\nLeast-connected community pairs ({args.method}) for {label}:")
    if df.empty:
        print("  (no candidate pairs — topic absent, or <2 communities clear the min-papers bar)")
    else:
        print(df.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
