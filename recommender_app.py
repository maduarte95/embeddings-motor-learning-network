"""
Streamlit UI for exploring least-connected community pairs within semantic topics.

Uses scoring functions from ``community_recommender``.
Pick a target input by **topic**, **paper**, or **author**; calls
the appropriate``recommend_for_*`` function and renders ranked pairs of communitieswith
human-readable topic keywords and community names.

Run:
    pixi run streamlit run recommender_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

import community_recommender as cr
from embedding_loaders import REGISTRY
from topic_store import doc_topics_path, load_topic_words

DATA_DIR = Path("data")
# Keywords-script output, same source build_semantic_web_data.py uses (so the
# UI's community labels match the web viz). The manual file is intentionally ignored.
COMMUNITY_NAMES = DATA_DIR / "community_names.json"


# ---------------------------------------------------------------------------
# Cached loaders (heavy objects built once, reused across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading citation graph…")
def load_graph() -> nx.DiGraph:
    return nx.read_graphml(cr.GRAPHML)


@st.cache_resource(show_spinner="Computing community connectedness…")
def get_connectedness(method: str) -> cr.Connectedness:
    # Connectedness is pure citation structure -> depends on method only.
    return cr.compute_connectedness(load_graph(), method=method)


@st.cache_resource(show_spinner="Mapping topics to communities…")
def get_membership(embedding: str) -> cr.TopicMembership:
    return cr.load_membership(load_graph(), embedding)


@st.cache_data(show_spinner=False)
def topic_keywords(embedding: str) -> dict[int, str]:
    """topic id -> comma-joined top keywords, for readable labels."""
    df = load_topic_words(embedding)
    return {
        int(r.topic_id): str(r.words).replace(" | ", ", ")
        for r in df.itertuples()
    }


@st.cache_data(show_spinner=False)
def community_names() -> dict[str, str]:
    """community id -> label, from the keywords-script `community_names.json`."""
    if not COMMUNITY_NAMES.exists():
        return {}
    return {str(k): str(v) for k, v in json.loads(COMMUNITY_NAMES.read_text()).items()}


@st.cache_data(show_spinner=False)
def paper_table() -> pd.DataFrame:
    """One row per node: id + title/authors/year/topic/cluster for display & search."""
    G = load_graph()
    rows = [
        {
            "node_id": str(n),
            "title": a.get("title") or a.get("label") or "",
            "authors": a.get("authors") or "",
            "year": a.get("year") or "",
            "journal": a.get("journal") or "",
            "cluster": str(a.get(cr.COMMUNITY_ATTR)) if a.get(cr.COMMUNITY_ATTR) is not None else "",
        }
        for n, a in G.nodes(data=True)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def available_embeddings() -> list[str]:
    """Registered models that actually have a topics table on disk."""
    return [k for k in REGISTRY if doc_topics_path(k).exists()] or list(REGISTRY)


def comm_label(cid: str) -> str:
    """Community name; falls back to the bare id for unnamed (<30-paper) communities."""
    return community_names().get(str(cid)) or str(cid)


def render_metric_explainer() -> None:
    """Collapsible sidebar panel explaining how each connectedness measure is computed."""
    with st.expander("How the metrics are computed"):
        st.markdown(
            "Each community gets a citation profile from the membership matrix "
            "$S$ ($S_{gi}=1$ iff paper $i$ is in community $g$) and the paper-cites-paper "
            "adjacency $A$. Every measure reduces to a community×community score; "
            "**lower = less connected**, so pairs are ranked ascending."
        )
        st.markdown("**Bibliographic coupling** — cosine of *reference* profiles $S A$:")
        st.latex(r"\text{coup}(a,b)=\cos\big(S A_a,\; S A_b\big)")
        st.caption("How much two communities cite the **same references** (Kessler 1963). "
                   "Cosine puts both sizes in the denominator, so it is size-normalised.")

        st.markdown("**Co-citation** — cosine of *cited-by* profiles $S A^\\top$:")
        st.latex(r"\text{cocit}(a,b)=\cos\big(S A^\top_a,\; S A^\top_b\big)")
        st.caption("How often two communities are **cited together** by later papers (Small 1973).")

        st.markdown("**Combined** — the mean of the two:")
        st.latex(r"\tfrac{1}{2}\big(\text{coup}+\text{cocit}\big)")

        st.markdown("**Conductance** — direct edges between the pair over the smaller volume:")
        st.latex(r"\phi(a,b)=\frac{e(a\!\to\!b)+e(b\!\to\!a)}{\min\big(\text{vol}(a),\,\text{vol}(b)\big)}")
        st.caption("$e(\\cdot)$ counts direct citations; $\\text{vol}$ is total community degree. "
                   "⚠️ Partner-size bias: large, prolific communities score low regardless of "
                   "whether they truly avoid each other — cross-check with bibliographic coupling.")

        st.divider()
        st.markdown(
            "**Candidate pairs.** Within a topic, only communities with "
            "**≥ *min papers*** papers in it qualify; all $\\binom{n}{2}$ pairs of qualifiers "
            "are scored and the bottom-*k* are shown. Community names come from "
            "`community_names.json` (TF-IDF keyword labels for communities with ≥30 papers; "
            "smaller ones show their numeric id)."
        )


def decorate_recs(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable community-name columns next to the raw ids."""
    if df.empty:
        return df
    out = df.copy()
    out.insert(out.columns.get_loc("community_a") + 1, "name_a",
               out["community_a"].map(comm_label))
    out.insert(out.columns.get_loc("community_b") + 1, "name_b",
               out["community_b"].map(comm_label))
    return out


def papers_in(community: str, topic: int, membership: cr.TopicMembership) -> pd.DataFrame:
    """Papers belonging to ``community`` AND ``topic`` (the overlap shown per side)."""
    ids = [
        n for n, c in membership.comm_by_node.items()
        if str(c) == str(community) and membership.topic_by_node.get(n) == topic
    ]
    tbl = paper_table().set_index("node_id")
    present = [i for i in ids if i in tbl.index]
    cols = ["title", "authors", "year", "journal"]
    return tbl.loc[present, cols].reset_index() if present else pd.DataFrame(columns=["node_id"] + cols)


def topic_breakdown(topic: int, membership: cr.TopicMembership, min_papers: int) -> pd.DataFrame:
    """Every community present in ``topic`` with its paper count and whether it qualifies."""
    part = membership.participation.get(topic, {})
    rows = [
        {
            "community": c,
            "name": community_names().get(str(c), ""),
            "papers_in_topic": n,
            "cleared_bar": n >= min_papers,
        }
        for c, n in part.items()
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("papers_in_topic", ascending=False).reset_index(drop=True) if not df.empty else df


def show_topic_breakdown(
    topic: int, membership: cr.TopicMembership, min_papers: int, keywords: str = ""
) -> None:
    """Collapsible panel: which communities are in a topic and which cleared the bar."""
    df = topic_breakdown(topic, membership, int(min_papers))
    n_present = len(df)
    n_cleared = int(df["cleared_bar"].sum()) if n_present else 0
    n_pairs = n_cleared * (n_cleared - 1) // 2
    title = f"Topic {topic}" + (f" — {keywords}" if keywords else "")
    label = (
        f"{title}  ·  {n_present} communities present · "
        f"{n_cleared} cleared the bar · {n_pairs} candidate pair(s)"
    )
    with st.expander(label):
        if keywords:
            st.markdown(f"**Topic {topic} keywords:** {keywords}")
        st.markdown(
            f"**Criteria.** A community *qualifies* if it has **≥ {int(min_papers)} "
            f"papers in this topic** (the *min papers* slider). All pairs of qualifying "
            f"communities are formed — `C({n_cleared}, 2) = {n_pairs}` — then ranked "
            f"ascending by connectedness (lower = less connected); the bottom-k are shown above. "
            f"Communities below the threshold are listed but excluded from pairing."
        )
        if df.empty:
            st.info("No communities recorded in this topic.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Least-connected topics explorer", layout="wide")
    st.title("Least-connected community pairs within topics")
    st.caption(
        "Within a semantic topic, surface citation communities that work on the "
        "same material but barely cite each other — same problem space, no cross-talk."
    )

    embeddings = available_embeddings()

    with st.sidebar:
        st.header("Parameters")
        embedding = st.selectbox("Embedding / topic model", embeddings)
        method = st.selectbox(
            "Connectedness measure", cr.METHODS,
            index=cr.METHODS.index(cr.DEFAULT_METHOD),
            help="Citation-space score; lower = less connected. Independent of the embedding.",
        )
        if method == "conductance":
            st.info(
                "Conductance has a partner-size bias: big, prolific communities "
                "rank as 'disconnected' regardless. Cross-check with bibliographic_coupling.",
                icon="⚠️",
            )
        k = st.number_input("Bottom-k pairs (0 = all)", min_value=0, value=cr.DEFAULT_K, step=1)
        min_papers = st.number_input(
            "Min papers a community needs in a topic", min_value=1,
            value=cr.DEFAULT_MIN_PAPERS, step=1,
        )
        render_metric_explainer()

    conn = get_connectedness(method)
    membership = get_membership(embedding)
    kw = topic_keywords(embedding)
    k_arg = None if k == 0 else int(k)

    mode = st.radio("Search by", ["Topic", "Paper", "Author", "All topics"], horizontal=True)

    recs = cr._empty_recs()
    subtitle = ""
    target_topics: list[int] = []   # topics to show the community breakdown for

    if mode == "Topic":
        topics = sorted(membership.participation)
        if not topics:
            st.warning("No non-outlier topics for this model.")
            return
        topic = st.selectbox(
            "Topic", topics,
            format_func=lambda t: f"{t} — {kw.get(t, '(no keywords)')}",
        )
        recs = cr.recommend_for_topic(topic, conn, membership, k=k_arg, min_papers=int(min_papers))
        subtitle = f"Topic {topic} — {kw.get(topic, '')}"
        target_topics = [topic]

    elif mode == "Paper":
        query = st.text_input("Search title (or paste a node id)", placeholder="e.g. visuomotor adaptation")
        node_id = None
        if query:
            tbl = paper_table()
            if query in set(tbl["node_id"]):
                node_id = query
            else:
                hits = tbl[tbl["title"].str.contains(query, case=False, na=False)].head(50)
                if hits.empty:
                    st.warning("No matching titles.")
                else:
                    pick = st.selectbox(
                        "Matching papers",
                        hits["node_id"].tolist(),
                        format_func=lambda nid: hits.set_index("node_id").loc[nid, "title"][:120],
                    )
                    node_id = pick
        if node_id:
            t = membership.topic_by_node.get(str(node_id))
            recs = cr.recommend_for_paper(node_id, conn, membership, k=k_arg, min_papers=int(min_papers))
            subtitle = f"Paper {node_id} — topic {t}: {kw.get(t, '')}"
            if t is not None and t != cr.OUTLIER_TOPIC:
                target_topics = [t]

    elif mode == "Author":
        query = st.text_input("Author name (substring)", placeholder="e.g. rizzolatti")
        if query:
            recs = cr.recommend_for_author(
                query, load_graph(), conn, membership, k=k_arg, min_papers=int(min_papers)
            )
            subtitle = f"Author '{query}' — across {recs['topic'].nunique() if not recs.empty else 0} topic(s)"
            if not recs.empty:
                target_topics = sorted(recs["topic"].unique())

    else:  # All topics — pool every topic's pairs and rank globally
        st.caption(
            "Scores are still computed **intra-topic** (each pair only against its own "
            "topic's communities); this view just stacks them and sorts globally. "
            "Coupling/co-citation are cosines in [0,1] so cross-topic ranking is fair; "
            "conductance is not normalised that way — read its global order with care."
        )
        if k_arg is not None:
            st.warning(
                f"Per-topic **k = {k_arg}** caps each topic to its {k_arg} worst pairs "
                "*before* pooling, so this ranking misses globally-low pairs from "
                "topics with many candidates. For a true global ranking set "
                "**Bottom-k pairs = 0** in the sidebar and let *Show top N* do the limiting.",
                icon="💡",
            )
        top_n = st.number_input("Show top N pairs globally", min_value=1, value=50, step=10)
        allrecs = cr.recommend_all_topics(conn, membership, k=k_arg, min_papers=int(min_papers))
        if not allrecs.empty:
            recs = (
                allrecs.sort_values(
                    "connectedness", ascending=True, kind="mergesort", na_position="last"
                )
                .head(int(top_n))
                .reset_index(drop=True)
            )
            recs.insert(1, "topic_keywords", recs["topic"].map(lambda t: kw.get(int(t), "")))
            subtitle = (
                f"Global ranking — {len(recs)} most-disconnected pairs "
                f"across {allrecs['topic'].nunique()} topics"
            )
            # No per-topic breakdown here: it could be dozens of panels.

    if subtitle:
        st.subheader(subtitle)

    # Community breakdown per target topic
    for t in target_topics:
        show_topic_breakdown(t, membership, int(min_papers), keywords=kw.get(t, ""))

    if recs.empty:
        st.info("No candidate pairs yet — choose a target above, or relax 'min papers'.")
        return

    st.dataframe(decorate_recs(recs), use_container_width=True, hide_index=True)

    # Pick a pair and see the papers on each side.
    st.divider()
    st.markdown("#### Inspect a pair")
    options = list(range(len(recs)))
    idx = st.selectbox(
        "Pair", options,
        format_func=lambda i: (
            f"topic {recs.iloc[i]['topic']} ({kw.get(int(recs.iloc[i]['topic']), '')[:40]}): "
            f"{comm_label(recs.iloc[i]['community_a'])}  ✗  {comm_label(recs.iloc[i]['community_b'])}  "
            f"(connectedness={recs.iloc[i]['connectedness']:.4f})"
        ),
    )
    row = recs.iloc[idx]
    topic = int(row["topic"])
    left, right = st.columns(2)
    for col, side in zip((left, right), ("a", "b")):
        cid = row[f"community_{side}"]
        with col:
            st.markdown(f"**{comm_label(cid)}** — {int(row[f'n_{side}_in_topic'])} papers in topic")
            st.dataframe(papers_in(cid, topic, membership), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
