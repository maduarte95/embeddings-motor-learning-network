"""
Build static assets for the interactive web version of the citation graph.

Reads the Gephi-exported graphml plus the existing per-cluster summary CSVs
and writes compact JSON / binary files into ``web/data/`` that the sigma.js
frontend in ``web/`` consumes.

Outputs (relative to project root):
    web/data/nodes.json       - per-paper records (positions, color, metadata)
    web/data/clusters.json    - cluster name, centroid, color, top-3 lists
    web/data/edges_out.bin    - directed CSR (out-neighbours) as uint32
    web/data/edges_in.bin     - directed CSR (in-neighbours) as uint32
    web/data/abstracts.json   - {node_id: abstract}, lazy-loaded by frontend
"""

import colorsys
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#these two are on invalid directories on purpose, they will be skipped
SUMMARIES_OUTPUT_DIR = Path().cwd() 
BRIDGE_OUTPUT_DIR = Path().cwd()

PROJECT_ROOT = Path().cwd()
COMMUNITY_ATTR = 'cluster'
DATA_DIR = Path('data')
COMMUNITY_NAMES_JSON = DATA_DIR / 'community_names.json'
GRAPHML_FILE = DATA_DIR / "citation_network_with_topics_new.graphml"

NS = {'g': 'http://graphml.graphdrawing.org/xmlns'}

WEB_DIR = PROJECT_ROOT / f"web_{COMMUNITY_ATTR}"
WEB_DATA_DIR = WEB_DIR / "data"
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

TOP_BRIDGE_PAPERS = 50  # how many bridge papers to flag with `bridge: 1`


# ── Graphml parsing ─────────────────────────────────────────────────────────

def parse_graphml():
    print(f"Parsing {GRAPHML_FILE} ...")
    tree = ET.parse(GRAPHML_FILE)
    root = tree.getroot()

    # Map graphml key id -> human attribute name
    keys = {}
    for key in root.findall('g:key', NS):
        keys[key.attrib['id']] = key.attrib['attr.name']

    nodes = []          # list of (graphml_id, attrs_dict)
    id_to_idx = {}      # graphml_id -> dense integer index

    for node_el in root.findall('g:graph/g:node', NS):
        nid = node_el.attrib['id']
        attrs = {}
        for d in node_el.findall('g:data', NS):
            name = keys.get(d.attrib['key'], d.attrib['key'])
            attrs[name] = d.text
        id_to_idx[nid] = len(nodes)
        nodes.append((nid, attrs))
    print(f"  {len(nodes)} nodes")

    edges = []  # list of (src_idx, tgt_idx)
    for edge_el in root.findall('g:graph/g:edge', NS):
        s = id_to_idx.get(edge_el.attrib['source'])
        t = id_to_idx.get(edge_el.attrib['target'])
        if s is not None and t is not None:
            edges.append((s, t))
    print(f"  {len(edges)} edges")

    return nodes, edges, id_to_idx


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def _hex_color(r, g, b):
    try:
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except (TypeError, ValueError):
        return "#cccccc"


def _make_palette(n):
    """Generate n visually-separated hex colors via golden-angle HSV hues.

    Evenly-spaced hues would give consecutively-numbered communities near-identical
    colors (and `cluster` ids are size-sorted, so the largest communities are
    consecutive). The golden-angle increment spreads consecutive ids far apart in
    hue, keeping neighbouring communities distinguishable for as long as possible.
    """
    golden = 0.61803398875
    palette = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * golden) % 1.0, 0.70, 0.95)
        palette.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return palette


# ── Build node payload ──────────────────────────────────────────────────────

def build_nodes(raw_nodes, bridge_ids, names_json):
    records = []
    abstracts = {}
    with open(names_json, 'r', encoding='utf-8') as f:
        clusters = {int(k): v for k, v in json.load(f).items()}
    # Color every named community from a generated palette rather than the graphml
    # r/g/b: Gephi only colored the top 20 communities and baked the rest grey, so
    # named communities 20+ would otherwise render grey. Unnamed communities stay grey.
    palette = _make_palette(len(clusters))
    cluster_color = {cid: palette[i] for i, cid in enumerate(sorted(clusters))}
    for nid, a in raw_nodes:
        cluster = _to_int(a.get(COMMUNITY_ATTR), -1)
        color = cluster_color.get(cluster, "#c0c0c0")
        if a.get('title') == 'Teaching the simple suture to medical students for long-term retention of skill':
            pass
        rec = {
            'id': nid,
            'title': (a.get('title') or '').strip(),
            'authors': (a.get('authors') or '').strip(),
            'keywords': (a.get('keywords') or '').strip(),
            'year': _to_int(a.get('year')),
            'journal': (a.get('journal') or '').strip(),
            'doi': (a.get('name') or '').strip(),
            'topic': (a.get('topic') or '').strip(),
            'cluster': cluster,
            'x': round(_to_float(a.get('x')), 2),
            'y': round(_to_float(a.get('y')), 2),
            'size': round(_to_float(a.get('size'), 1.0), 3),
            'color': color,
            'indegree': _to_int(a.get('Eingangsgrad'), 0),
            'degree': _to_int(a.get('Grad'), 0),
        }
        if nid in bridge_ids:
            rec['bridge'] = 1
        records.append(rec)
        abstract = (a.get('abstract') or '').strip()
        if abstract:
            abstracts[nid] = abstract
    return records, abstracts


# ── Build cluster payload ───────────────────────────────────────────────────

def parse_community_summary(path):
    """Pivot community_summary.csv into {cluster_id: {top_papers, top_authors, top_keywords}}."""
    try:
        df = pd.read_csv(path)
    except:
        print(f'Warning: community summary not found at {path}')
        df = pd.DataFrame()
    out = defaultdict(lambda: {'top_papers': [], 'top_authors': [], 'top_keywords': []})
    for _, row in df.iterrows():
        cid = int(row['community'])
        category = row['category']
        name = row['name']
        detail = row['detail'] if isinstance(row['detail'], str) else ''
        if category == 'paper':
            indeg = re.search(r'in-degree=(\d+)', detail)
            year = re.search(r'year=(\d+)', detail)
            out[cid]['top_papers'].append({
                'title': name,
                'in_degree': int(indeg.group(1)) if indeg else None,
                'year': int(year.group(1)) if year else None,
            })
        elif category == 'author':
            papers = re.search(r'papers=(\d+)', detail)
            out[cid]['top_authors'].append({
                'name': name,
                'papers': int(papers.group(1)) if papers else None,
            })
        elif category == 'keyword':
            tfidf = re.search(r'tfidf=([\d.]+)', detail)
            out[cid]['top_keywords'].append({
                'keyword': name,
                'tfidf': float(tfidf.group(1)) if tfidf else None,
            })
    return out


def build_clusters(node_records, summary_path, names_json, clustering_column):
    # Per-cluster aggregation: weighted centroid, dominant color, node count
    sums_x = defaultdict(float)
    sums_y = defaultdict(float)
    sums_w = defaultdict(float)
    counts = Counter()
    color_votes = defaultdict(Counter)

    for r in node_records:
        cid = r[clustering_column]
        if isinstance(cid, str):
            old_cid = cid
            cid = int(cid)
            print(f"Casting {old_cid} resulted in {cid}")
        # if cid < 0:
        #     continue
        w = max(r['size'], 0.1)
        sums_x[cid] += r['x'] * w
        sums_y[cid] += r['y'] * w
        sums_w[cid] += w
        counts[cid] += 1
        color_votes[cid][r['color']] += 1

    summary = parse_community_summary(summary_path)
    with open(names_json, 'r', encoding='utf-8') as f:
        names = {int(k): v for k, v in json.load(f).items()}

    # Only emit clusters that have a human-readable name (the top 30 from
    # extract_names.py). Hundreds of tiny Leiden clusters exist in the graphml
    # but they aren't useful as labels.
    clusters = {}
    for cid in sorted(names.keys()):
        if cid not in counts:
            continue
        n = counts[cid]
        cx = sums_x[cid] / sums_w[cid]
        cy = sums_y[cid] / sums_w[cid]
        clusters[str(cid)] = {
            'id': cid,
            'name': names.get(cid, f'Cluster {cid}'),
            'color': color_votes[cid].most_common(1)[0][0],
            'centroid': [round(cx, 2), round(cy, 2)],
            'size': n,
            'top_papers': summary.get(cid, {}).get('top_papers', []),
            'top_authors': summary.get(cid, {}).get('top_authors', []),
            'top_keywords': summary.get(cid, {}).get('top_keywords', []),
        }
    return clusters


# ── Build CSR adjacency files ───────────────────────────────────────────────

def build_csr(num_nodes, edges, direction):
    """
    Build a CSR adjacency for the requested direction.
        direction == 'out': for each node, list of nodes it cites (target idx)
        direction == 'in':  for each node, list of nodes that cite it (source idx)
    Returns (offsets: list[int], targets: list[int]).
    """
    buckets = [[] for _ in range(num_nodes)]
    if direction == 'out':
        for s, t in edges:
            buckets[s].append(t)
    elif direction == 'in':
        for s, t in edges:
            buckets[t].append(s)
    else:
        raise ValueError(direction)

    offsets = [0]
    targets = []
    for b in buckets:
        targets.extend(b)
        offsets.append(len(targets))
    return offsets, targets


def write_csr(path, offsets, targets):
    """
    Binary CSR layout (little-endian uint32):
        [N]
        [offsets: N+1 entries]
        [targets: total entries]
    """
    n = len(offsets) - 1
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', n))
        f.write(struct.pack('<{}I'.format(len(offsets)), *offsets))
        f.write(struct.pack('<{}I'.format(len(targets)), *targets))


# ── Bridge papers ───────────────────────────────────────────────────────────

def load_bridge_ids(path, top_n):
    if not path.exists():
        print(f"  WARN: {path} not found, skipping bridge flag")
        return set()
    df = pd.read_csv(path)
    return set(df.head(top_n)['node_id'].astype(str).tolist())


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    bridge_ids = load_bridge_ids(BRIDGE_OUTPUT_DIR / "bridge_papers.csv", TOP_BRIDGE_PAPERS)
    print(f"Flagged {len(bridge_ids)} bridge papers")


    raw_nodes, edges, id_to_idx = parse_graphml()
    node_records, abstracts = build_nodes(raw_nodes, bridge_ids, COMMUNITY_NAMES_JSON)

    years = [r['year'] for r in node_records if r['year'] is not None]
    year_min = min(years) if years else None
    year_max = max(years) if years else None

    clusters = build_clusters(
        node_records,
        SUMMARIES_OUTPUT_DIR / "community_summary.csv",
        COMMUNITY_NAMES_JSON,
        clustering_column='cluster'
    )

    print(f"Built {len(clusters)} cluster summaries")

    out_offsets, out_targets = build_csr(len(node_records), edges, 'out')
    in_offsets, in_targets = build_csr(len(node_records), edges, 'in')

    # ── Write outputs ──────────────────────────────────────────────────────
    nodes_payload = {
        'year_min': year_min,
        'year_max': year_max,
        'nodes': node_records,
    }
    nodes_path = WEB_DATA_DIR / "nodes.json"
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(nodes_payload, f, ensure_ascii=False, separators=(',', ':'))

    clusters_path = WEB_DATA_DIR / "clusters.json"
    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, ensure_ascii=False, separators=(',', ':'))


    abstracts_path = WEB_DATA_DIR / "abstracts.json"
    with open(abstracts_path, 'w', encoding='utf-8') as f:
        json.dump(abstracts, f, ensure_ascii=False, separators=(',', ':'))

    out_path = WEB_DATA_DIR / "edges_out.bin"
    in_path = WEB_DATA_DIR / "edges_in.bin"
    write_csr(out_path, out_offsets, out_targets)
    write_csr(in_path, in_offsets, in_targets)

    # ── Sanity report ──────────────────────────────────────────────────────
    def mb(p):
        return f"{p.stat().st_size / (1024 * 1024):.2f} MB"

    print("\n-- Build summary --------------------------------------------")
    print(f"nodes        : {len(node_records):>7,}  ->  {mb(nodes_path):>9}  ({nodes_path.name})")
    print(f"clusters     : {len(clusters):>7,}  ->  {mb(clusters_path):>9}  ({clusters_path.name})")
    print(f"abstracts    : {len(abstracts):>7,}  ->  {mb(abstracts_path):>9}  ({abstracts_path.name})")
    print(f"out-edges    : {len(edges):>7,}  ->  {mb(out_path):>9}  ({out_path.name})")
    print(f"in-edges     : {len(edges):>7,}  ->  {mb(in_path):>9}  ({in_path.name})")
    print(f"year range   : {year_min} - {year_max}")

    # Cross-check from the plan
    n2 = next((r for r in node_records if r['id'] == 'n2'), None)
    if n2:
        print(f"\nSanity (n2): cluster={n2['cluster']}, in-degree={n2['indegree']}, year={n2['year']}")
        print(f"             title={n2['title'][:60]}...")
    if '0' in clusters:
        kws = [k['keyword'] for k in clusters['0']['top_keywords']]
        print(f"Sanity (cluster 0 keywords): {kws}")


if __name__ == '__main__':
    main()
