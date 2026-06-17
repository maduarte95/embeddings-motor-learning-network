"""
Driver: compute embedding-vs-citation NN Jaccard overlap on the motor-learning corpus.

For each citation-space method (bibliographic_coupling, co_citation, combined):
  - Compute per-paper Jaccard against embedding-space cosine NN.
  - Aggregate overall, by BERTopic topic, and by Leiden community.
  - Compare against a label-permutation null (citation-NN lists are
    reassigned to papers at random, preserving the marginal list-size
    distribution) to gauge how far real Jaccards sit from chance.

Outputs are written to data/overlap_results/.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from citation_neighbors import nearest_neighbors as citation_nn
from embedding_loaders import DEFAULT_EMBEDDING
from embedding_neighbors import cosine_nearest_neighbors
from embedding_store import load_embeddings
from topic_store import load_doc_topics
from jaccard_overlap import aggregate_jaccard, per_paper_jaccard, shuffle_null


DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "overlap_results"
GRAPHML = DATA_DIR / "citation_network_with_topics_new.graphml"

K = 10
CITATION_METHODS = ["bibliographic_coupling", "co_citation", "combined"]
COMMUNITY_ATTR = "cluster"
NULL_RUNS = 100


def load_data(
    embedding_key: str = DEFAULT_EMBEDDING,
) -> tuple[nx.DiGraph, np.ndarray, list[str], dict[str, int], dict[str, str]]:
    print(f"Loading graph from {GRAPHML}...")
    G = nx.read_graphml(GRAPHML)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"Loading embeddings (model: {embedding_key})...")
    # node_ids travel with the vectors — alignment is by id, not row order.
    emb, node_ids, emb_meta = load_embeddings(embedding_key)
    print(f"  shape={emb.shape}  model={emb_meta.get('model')}")

    print(f"Loading document-topic map (model: {embedding_key})...")
    df = load_doc_topics(embedding_key)
    topic_by_node = dict(zip(df["node_id"].astype(str), df["topic"]))
    print(f"  {len(df):,} rows")

    community_by_node: dict[str, str] = {}
    for n, attrs in G.nodes(data=True):
        c = attrs.get(COMMUNITY_ATTR)
        if c is not None:
            community_by_node[n] = str(c)
    print(f"  community attribute '{COMMUNITY_ATTR}': "
          f"{len(community_by_node):,} nodes assigned, "
          f"{len(set(community_by_node.values()))} distinct communities")

    return G, emb, node_ids, topic_by_node, community_by_node


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding", default=DEFAULT_EMBEDDING,
        help=f"Embedding model key (default: {DEFAULT_EMBEDDING}).",
    )
    args = parser.parse_args()

    G, emb, node_ids, topic_by_node, community_by_node = load_data(args.embedding)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nComputing embedding NN (k={K})...")
    nn_emb = cosine_nearest_neighbors(emb, node_ids, k=K)

    summary_rows = []
    for method in CITATION_METHODS:
        print(f"\n=== {method} ===")
        nn_cit = citation_nn(G, method=method, k=K)
        jacc, excluded = per_paper_jaccard(nn_emb, nn_cit, k=K, strict=True)
        print(f"  papers included: {len(jacc):,}  |  excluded (short list): {len(excluded):,}")

        if not jacc:
            print("  no papers met the strict-k requirement; skipping aggregations")
            continue

        agg_overall   = aggregate_jaccard(jacc, groups=None)
        agg_topic     = aggregate_jaccard(jacc, groups=topic_by_node, group_name="topic")
        agg_community = aggregate_jaccard(jacc, groups=community_by_node, group_name="community")

        agg_overall.to_csv(OUT_DIR / f"{method}_overall.csv", index=False)
        agg_topic.to_csv(OUT_DIR / f"{method}_by_topic.csv", index=False)
        agg_community.to_csv(OUT_DIR / f"{method}_by_community.csv", index=False)
        pd.DataFrame(
            {"node_id": list(jacc.keys()), "jaccard": list(jacc.values())}
        ).to_csv(OUT_DIR / f"{method}_per_paper.csv", index=False)

        real_mean = float(agg_overall.loc[0, "mean"])
        real_median = float(agg_overall.loc[0, "median"])
        print(f"  overall  mean={real_mean:.4f}  median={real_median:.4f}  n={int(agg_overall.loc[0, 'n'])}")

        print(f"  computing null baseline ({NULL_RUNS} shuffles)...")
        null_df = shuffle_null(nn_emb, nn_cit, k=K, n_runs=NULL_RUNS, seed=0)
        null_mean = null_df["mean"].mean()
        null_std = null_df["mean"].std()
        null_df.to_csv(OUT_DIR / f"{method}_null.csv", index=False)
        print(f"  null     mean={null_mean:.4f}  std={null_std:.4f}")
        print(f"  effect size (real - null)/null_std = {(real_mean - null_mean) / null_std:.2f}")

        # Topic / community extremes for a quick eyeball
        topic_top    = agg_topic.head(5)
        topic_bottom = agg_topic.tail(5)
        print("\n  highest-overlap topics:")
        print(topic_top.to_string(index=False, float_format="{:.4f}".format))
        print("\n  lowest-overlap topics:")
        print(topic_bottom.to_string(index=False, float_format="{:.4f}".format))

        summary_rows.append({
            "method": method,
            "n_included": len(jacc),
            "n_excluded": len(excluded),
            "real_mean": real_mean,
            "real_median": real_median,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": (real_mean - null_mean) / null_std if null_std > 0 else np.nan,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
        print("\n=== Summary ===")
        print(summary_df.to_string(index=False, float_format="{:.4f}".format))
        print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
