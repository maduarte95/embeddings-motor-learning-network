"""
Build static assets for the *semantic* web viz (~/motor-semantic-viz).

Sibling of ``build_web_data.py``. Same output schema and the same sigma.js
frontend, with two differences:

  1. Node x/y come from a 2D UMAP projection of the embeddings
     (``data/embeddings/{key}.npz`` via embedding_store; default SPECTER2)
     instead of the Gephi citation layout.
  2. Two coexisting groupings are emitted, each with its own colors,
     centroids and legend payload:
        - topics.json      BERTopic topics      (default color)
        - communities.json Leiden communities   (the graphml ``cluster`` attr)

Every node record carries *both* a ``topic`` and ``cluster`` id and a
precomputed color for each, so the frontend can switch "Color by" and mute
either grouping independently without recomputing anything.

Citation edges (edges_out.bin / edges_in.bin) and abstracts.json are built
exactly as in build_web_data.py — the layout changed, the citation graph did
not.

Outputs (relative to project root), into ``web_semantic/data/``:
    nodes.json        per-paper records (UMAP x/y, topic/cluster + colors, meta)
    topics.json       BERTopic topic name, color, UMAP centroid, size, top words
    communities.json  Leiden community name, color, UMAP centroid, size
    edges_out.bin     directed CSR (out-neighbours) as uint32
    edges_in.bin      directed CSR (in-neighbours) as uint32
    abstracts.json    {node_id: abstract}, lazy-loaded by frontend

Run:  pixi run python build_semantic_web_data.py [--recompute]
"""

import argparse
import colorsys
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_loaders import DEFAULT_EMBEDDING
from embedding_store import load_embeddings
from topic_store import topic_words_path, load_doc_topics

# ── Paths / config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

GRAPHML_FILE = DATA_DIR / "citation_network_with_topics_new.graphml"
COMMUNITY_NAMES_JSON = DATA_DIR / "community_names.json"

WEB_DIR = PROJECT_ROOT / "web_semantic"
WEB_DATA_DIR = WEB_DIR / "data"


def umap_cache_path(key):
    """2D-UMAP cache for embedding ``key`` (each model has its own layout)."""
    return DATA_DIR / f"umap_2d_{key}.npz"


COMMUNITY_ATTR = "cluster"   # Leiden community attribute on graphml nodes
# Per-node BERTopic topic comes from document_topics_{key}.csv (topic_store),
# not the graphml 'topic' attr, so each embedding key stays self-consistent.
MIN_GROUP_SIZE = 30          # only name/colour/legend groups at least this big
TOP_BRIDGE_PAPERS = 50       # how many bridge papers to flag with `bridge: 1`
BRIDGE_CSV = PROJECT_ROOT / "bridge_papers.csv"  # optional; skipped if absent

# UMAP layout params. cosine matches how SPECTER2 vectors are compared in
# embedding_neighbors.py; the fixed seed keeps the map stable across rebuilds.
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_SEED = 42
UMAP_SPREAD = 12.0   # scale 2D coords into a comfortable sigma.js range

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}


# ── Small helpers ───────────────────────────────────────────────────────────

def _to_int(value, default=None):
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_palette(n):
    """n visually-separated hex colors via golden-angle HSV hues.

    Size-sorted ids are consecutive, so evenly-spaced hues would make the
    largest groups near-identical. The golden-angle increment spreads
    consecutive ids far apart in hue.
    """
    golden = 0.61803398875
    palette = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * golden) % 1.0, 0.70, 0.95)
        palette.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return palette


# ── 1. UMAP 2D layout (cached) ──────────────────────────────────────────────

def _embed_fingerprint(emb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(emb.shape).encode())
    h.update(np.ascontiguousarray(emb, dtype=np.float32).tobytes())
    return h.hexdigest()


def compute_umap_layout(recompute=False, embedding_key=DEFAULT_EMBEDDING):
    """Return {node_id: (x, y)} from a 2D UMAP of the cached embeddings.

    Embeddings and their node_ids are loaded together from embedding_store, so
    rows are aligned by id rather than by trusting a separate file's row order.
    """
    emb, node_ids, _ = load_embeddings(embedding_key)
    emb = emb.astype(np.float32)

    umap_cache = umap_cache_path(embedding_key)
    fp = _embed_fingerprint(emb)
    if not recompute and umap_cache.exists():
        cached = np.load(umap_cache, allow_pickle=True)
        if str(cached["fingerprint"]) == fp:
            print(f"  UMAP cache hit — loading {umap_cache.name}")
            coords = cached["coords"]
            ids = cached["node_ids"].astype(str)
            return dict(zip(ids.tolist(), coords))

    print(f"  Computing 2D UMAP on {emb.shape} (metric={UMAP_METRIC}, seed={UMAP_SEED})...")
    from umap import UMAP  # imported lazily; heavy dependency

    coords = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
    ).fit_transform(emb)

    # Center and scale to a stable, frontend-friendly coordinate range.
    coords = coords - coords.mean(axis=0)
    span = np.abs(coords).max()
    if span > 0:
        coords = coords / span * UMAP_SPREAD
    coords = coords.astype(np.float32)

    np.savez_compressed(
        umap_cache,
        coords=coords,
        node_ids=np.array(node_ids, dtype=object),
        fingerprint=fp,
    )
    print(f"  UMAP saved to {umap_cache.name}")
    return dict(zip(node_ids, coords))


# ── 2. Graphml parsing (kept index-aligned with the citation CSR) ───────────

def parse_graphml():
    print(f"Parsing {GRAPHML_FILE.name} ...")
    root = ET.parse(GRAPHML_FILE).getroot()
    keys = {k.attrib["id"]: k.attrib["attr.name"] for k in root.findall("g:key", NS)}

    nodes = []
    id_to_idx = {}
    for node_el in root.findall("g:graph/g:node", NS):
        nid = node_el.attrib["id"]
        attrs = {}
        for d in node_el.findall("g:data", NS):
            attrs[keys.get(d.attrib["key"], d.attrib["key"])] = d.text
        id_to_idx[nid] = len(nodes)
        nodes.append((nid, attrs))
    print(f"  {len(nodes)} nodes")

    edges = []
    for edge_el in root.findall("g:graph/g:edge", NS):
        s = id_to_idx.get(edge_el.attrib["source"])
        t = id_to_idx.get(edge_el.attrib["target"])
        if s is not None and t is not None:
            edges.append((s, t))
    print(f"  {len(edges)} edges")
    return nodes, edges, id_to_idx


# ── 3. Colour maps for both groupings ───────────────────────────────────────

def build_color_maps(raw_nodes, community_names, topic_by_node):
    """Return (community_color, topic_color, topic_sizes) dicts.

    Communities are coloured from their named set (community_names.json), like
    build_web_data. Topics are coloured for every topic with >= MIN_GROUP_SIZE
    papers; smaller topics and the -1 outlier bucket stay grey. Per-node topics
    come from the selected model's document_topics file (topic_by_node).
    """
    topic_sizes = Counter()
    for nid, _a in raw_nodes:
        topic_sizes[topic_by_node.get(nid, -1)] += 1

    community_ids = sorted(int(k) for k in community_names)
    comm_palette = _make_palette(len(community_ids))
    community_color = {cid: comm_palette[i] for i, cid in enumerate(community_ids)}

    big_topics = sorted(t for t, n in topic_sizes.items() if t != -1 and n >= MIN_GROUP_SIZE)
    topic_palette = _make_palette(len(big_topics))
    topic_color = {tid: topic_palette[i] for i, tid in enumerate(big_topics)}

    return community_color, topic_color, topic_sizes


# ── 4. Node payload ─────────────────────────────────────────────────────────

GREY = "#c0c0c0"


def build_nodes(raw_nodes, umap_xy, community_color, topic_color, bridge_ids,
                topic_by_node):
    records = []
    abstracts = {}
    missing_xy = 0
    for nid, a in raw_nodes:
        cluster = _to_int(a.get(COMMUNITY_ATTR), -1)
        topic = topic_by_node.get(nid, -1)
        xy = umap_xy.get(nid)
        if xy is None:
            missing_xy += 1
            x = y = 0.0
        else:
            x, y = float(xy[0]), float(xy[1])
        rec = {
            "id": nid,
            "title": (a.get("title") or "").strip(),
            "authors": (a.get("authors") or "").strip(),
            "keywords": (a.get("keywords") or "").strip(),
            "year": _to_int(a.get("year")),
            "journal": (a.get("journal") or "").strip(),
            "doi": (a.get("name") or "").strip(),
            "cluster": cluster,
            "topic": topic,
            "cluster_color": community_color.get(cluster, GREY),
            "topic_color": topic_color.get(topic, GREY),
            "x": round(x, 3),
            "y": round(y, 3),
            "size": round(_to_float(a.get("size"), 1.0), 3),
            "indegree": _to_int(a.get("Eingangsgrad"), 0),
            "degree": _to_int(a.get("Grad"), 0),
        }
        if nid in bridge_ids:
            rec["bridge"] = 1
        records.append(rec)
        abstract = (a.get("abstract") or "").strip()
        if abstract:
            abstracts[nid] = abstract
    if missing_xy:
        print(f"  WARN: {missing_xy} nodes had no UMAP position (placed at origin)")
    return records, abstracts


# ── 5. Grouping payloads (centroids computed in UMAP space) ─────────────────

def _centroids(node_records, group_key):
    sums_x, sums_y, sums_w = defaultdict(float), defaultdict(float), defaultdict(float)
    counts = Counter()
    for r in node_records:
        gid = r[group_key]
        w = max(r["size"], 0.1)
        sums_x[gid] += r["x"] * w
        sums_y[gid] += r["y"] * w
        sums_w[gid] += w
        counts[gid] += 1
    return sums_x, sums_y, sums_w, counts


def build_communities(node_records, community_names, community_color):
    sums_x, sums_y, sums_w, counts = _centroids(node_records, "cluster")
    out = {}
    for cid in sorted(int(k) for k in community_names):
        if cid not in counts:
            continue
        out[str(cid)] = {
            "id": cid,
            "name": community_names[str(cid)],
            "color": community_color.get(cid, GREY),
            "centroid": [round(sums_x[cid] / sums_w[cid], 3), round(sums_y[cid] / sums_w[cid], 3)],
            "size": counts[cid],
            "top_papers": [],
            "top_authors": [],
            "top_keywords": [],
        }
    return out


def _topic_names(embedding_key=DEFAULT_EMBEDDING):
    """{topic_id: 'word1 · word2 · word3'} from the model's topic_words file."""
    tw_path = topic_words_path(embedding_key)
    try:
        tw = pd.read_csv(tw_path)
    except FileNotFoundError:
        print(f"  WARN: {tw_path.name} not found, topics will be unnamed")
        return {}
    names = {}
    for _, row in tw.iterrows():
        words = [w.strip() for w in str(row["words"]).split("|")][:3]
        names[int(row["topic_id"])] = " · ".join(w for w in words if w)
    return names


def build_topics(node_records, topic_color, topic_names):
    sums_x, sums_y, sums_w, counts = _centroids(node_records, "topic")
    out = {}
    for tid in sorted(topic_color):  # only the named (>= MIN_GROUP_SIZE) topics
        if tid not in counts:
            continue
        out[str(tid)] = {
            "id": tid,
            "name": topic_names.get(tid, f"Topic {tid}"),
            "color": topic_color[tid],
            "centroid": [round(sums_x[tid] / sums_w[tid], 3), round(sums_y[tid] / sums_w[tid], 3)],
            "size": counts[tid],
            "top_words": topic_names.get(tid, ""),
        }
    return out


# ── 6. CSR citation edges (identical to build_web_data) ─────────────────────

def build_csr(num_nodes, edges, direction):
    buckets = [[] for _ in range(num_nodes)]
    if direction == "out":
        for s, t in edges:
            buckets[s].append(t)
    elif direction == "in":
        for s, t in edges:
            buckets[t].append(s)
    else:
        raise ValueError(direction)
    offsets, targets = [0], []
    for b in buckets:
        targets.extend(b)
        offsets.append(len(targets))
    return offsets, targets


def write_csr(path, offsets, targets):
    n = len(offsets) - 1
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<{}I".format(len(offsets)), *offsets))
        f.write(struct.pack("<{}I".format(len(targets)), *targets))


def load_bridge_ids(path, top_n):
    if not path.exists():
        print(f"  (no {path.name}; skipping bridge flag)")
        return set()
    df = pd.read_csv(path)
    return set(df.head(top_n)["node_id"].astype(str).tolist())


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="Ignore the UMAP cache.")
    ap.add_argument(
        "--embedding", default=DEFAULT_EMBEDDING,
        help=f"Embedding model key (default: {DEFAULT_EMBEDDING}).",
    )
    args = ap.parse_args()

    # Per-key outputs (model-dependent: layout, topics, community centroids) go
    # in a subdir; truly shared outputs (citations, abstracts) stay top-level.
    key_dir = WEB_DATA_DIR / args.embedding
    key_dir.mkdir(parents=True, exist_ok=True)

    print("Computing UMAP layout...")
    umap_xy = compute_umap_layout(recompute=args.recompute, embedding_key=args.embedding)

    raw_nodes, edges, _ = parse_graphml()

    with open(COMMUNITY_NAMES_JSON, encoding="utf-8") as f:
        community_names = json.load(f)

    # Per-node topic from the selected model's document_topics file.
    doc = load_doc_topics(args.embedding)
    topic_by_node = {str(n): _to_int(t, -1) for n, t in zip(doc["node_id"], doc["topic"])}

    community_color, topic_color, topic_sizes = build_color_maps(
        raw_nodes, community_names, topic_by_node
    )
    bridge_ids = load_bridge_ids(BRIDGE_CSV, TOP_BRIDGE_PAPERS)

    node_records, abstracts = build_nodes(
        raw_nodes, umap_xy, community_color, topic_color, bridge_ids, topic_by_node
    )

    communities = build_communities(node_records, community_names, community_color)
    topics = build_topics(node_records, topic_color, _topic_names(args.embedding))

    years = [r["year"] for r in node_records if r["year"] is not None]
    nodes_payload = {
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "nodes": node_records,
    }

    out_offsets, out_targets = build_csr(len(node_records), edges, "out")
    in_offsets, in_targets = build_csr(len(node_records), edges, "in")

    # ── Write outputs ───────────────────────────────────────────────────────
    def dump(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        return path

    # Per-key (model-dependent): node layout, topic grouping, and community
    # centroids (centroids live in this model's UMAP space).
    nodes_path = dump(key_dir / "nodes.json", nodes_payload)
    topics_path = dump(key_dir / "topics.json", topics)
    comm_path = dump(key_dir / "communities.json", communities)
    # Shared (model-independent): citation edges and abstracts.
    abstracts_path = dump(WEB_DATA_DIR / "abstracts.json", abstracts)
    out_path = WEB_DATA_DIR / "edges_out.bin"
    in_path = WEB_DATA_DIR / "edges_in.bin"
    write_csr(out_path, out_offsets, out_targets)
    write_csr(in_path, in_offsets, in_targets)

    def mb(p):
        return f"{p.stat().st_size / (1024 * 1024):.2f} MB"

    print("\n-- Build summary --------------------------------------------")
    print(f"nodes        : {len(node_records):>7,}  ->  {mb(nodes_path):>9}  ({nodes_path.name})")
    print(f"topics       : {len(topics):>7,}  ->  {mb(topics_path):>9}  ({topics_path.name})")
    print(f"communities  : {len(communities):>7,}  ->  {mb(comm_path):>9}  ({comm_path.name})")
    print(f"abstracts    : {len(abstracts):>7,}  ->  {mb(abstracts_path):>9}  ({abstracts_path.name})")
    print(f"edges        : {len(edges):>7,}  ->  {mb(out_path):>9}  ({out_path.name})")
    print(f"outliers     : topic -1 has {topic_sizes.get(-1, 0):,} papers (grey, no legend)")


if __name__ == "__main__":
    main()
