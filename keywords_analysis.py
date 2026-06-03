import re
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy.sparse
from typing import Final
from pathlib import Path

DATA_DIR = Path('data')

SYNONYMS_FILE = DATA_DIR / "keyword_synonyms_0.99_with_transitivity.json"
GRAPHML_FILE = DATA_DIR / "citation_network_with_topics_new.graphml"
PARQUET_FILE = GRAPHML_FILE.with_suffix(".parquet")
COMMUNITY_ATTR = 'cluster'
TOP_N_COMMUNITIES = 30

NORM: Final = 'l2'
IDF_BIAS: Final = 0.0
SYNONYMS_THRESHOLD: Final = 0.99
citation_network_df = pd.read_parquet(PARQUET_FILE)
TFIDF_OUTPUT_DIR: Final = Path().cwd() / "tf_idf_results"
TFIDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep only the TOP_N_CLUSTERS largest clusters by number of papers
if COMMUNITY_ATTR in citation_network_df.columns:
    cluster_sizes = citation_network_df[COMMUNITY_ATTR].dropna().value_counts()
    top_clusters = cluster_sizes.head(TOP_N_COMMUNITIES).index.tolist()
else:
    top_clusters = list(range(TOP_N_COMMUNITIES))

PALETTE_20 = [
    '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#000000',
]
MODULARITY_META = {f'{i}': {"label": f"Community {i}", "color": PALETTE_20[i % 20]} for i in range(TOP_N_COMMUNITIES)}


def correct_tfidf(X: scipy.sparse.csr_matrix, vectorizer: TfidfVectorizer, norm: str = None):
    """
    Correct sklearn's IDF formula and optionally re-normalize.

    sklearn computes idf as: 1 + log((1+n)/(1+df))  (the leading 1 is the bias we remove)
    We want:               IDF_BIAS + log((1+n)/(1+df))

    Because the vectorizer was fit with norm=None, X contains raw TF * wrong_idf,
    so we can safely divide out wrong_idf, apply corrected_idf, then normalize.
    """
    X_array = X.toarray()
    wrong_idf = vectorizer.idf_
    # Recover raw TF (safe because vectorizer was fit with norm=None)
    tf = X_array / wrong_idf
    corrected_idf = wrong_idf - 1.0 + IDF_BIAS
    tfidf = tf * corrected_idf

    if norm:
        tfidf = normalize(tfidf, norm=norm)

    return scipy.sparse.csr_matrix(tfidf)


def normalize_keyword(keyword: str) -> str:
    """Normalize a keyword to lowercase."""
    return keyword.lower()


def split_keywords(keywords_value):
    """Split a keywords value into individual keywords.
    Handles both pipe-separated strings and array-like values (e.g. from parquet)."""
    if keywords_value is None:
        return []

    # If it's already a list or numpy array, return as a flat list of strings
    if isinstance(keywords_value, (list, np.ndarray)):
        return [str(k).strip() for k in keywords_value if str(k).strip()]

    # Scalar NaN check
    try:
        if pd.isna(keywords_value):
            return []
    except (ValueError, TypeError):
        pass

    s = str(keywords_value).strip()
    if not s:
        return []

    # Split by '|'
    parts = [p.strip() for p in s.split('|') if p.strip()]
    return parts


def load_synonym_data(json_path: Path) -> dict:
    """Load the synonym dictionary from a JSON file."""
    if not json_path.exists():
        print(f"Error: Synonym dictionary file not found at {json_path}. Please create it.")
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            synonym_dict = json.load(f)
        print(f"Loaded {len(synonym_dict)} synonym groups from {json_path.name}.")
        return synonym_dict
    except Exception as e:
        print(f"Error reading or processing {json_path}: {e}")
        return {}


def load_synonym_map(synonym_dict: dict) -> dict:
    """
    Creates a canonical keyword map where all synonyms point to a single
    chosen representative (the normalized name of the original dictionary key).
    The keys and values in the map are normalized (lowercase).
    """
    canonical_map = {}

    for key, values in synonym_dict.items():
        canonical_name = normalize_keyword(key)
        all_variants = [key] + values

        for variant in all_variants:
            norm_variant = normalize_keyword(variant)
            if norm_variant not in canonical_map:
                canonical_map[norm_variant] = canonical_name

    return canonical_map


def calculate_canonical_tfidf(citation_network_df: pd.DataFrame, synonym_dict_path: Path):
    # 1. Filtering
    df = citation_network_df.copy()
    df = df.drop(df[df['keywords'].isin(["Unknown keywords"])].index)
    df = df[df[COMMUNITY_ATTR].isin(MODULARITY_META)]
    df = df.dropna().reset_index(drop=True)

    # 2. Load and Prepare Synonym Map
    print(f"Loading synonym data from: {synonym_dict_path}")
    synonym_data = load_synonym_data(synonym_dict_path)
    synonym_data['Purkinje Cell'].extend(['Purkinje Cell ( PC )'])
    synonym_map = load_synonym_map(synonym_data)

    # 3. Initialize QA Log
    qa_log = {}

    # 4. Rewrite Corpus using Canonical Keywords
    print("Rewriting corpus using canonical keywords...")
    canonical_corpus = {}
    cluster_paper_counts = Counter()
    all_canonical_keywords = set()

    for raw_keywords_str, modularity_class in zip(df['keywords'], df[COMMUNITY_ATTR]):
        if modularity_class not in canonical_corpus:
            canonical_corpus[modularity_class] = []

        terms = split_keywords(raw_keywords_str)

        cluster_paper_counts[modularity_class] += 1

        for raw_term in terms:
            norm_term = normalize_keyword(raw_term)
            canonical_term = synonym_map.get(norm_term, norm_term)
            all_canonical_keywords.add(canonical_term)
            canonical_corpus[modularity_class].append(canonical_term)

            # QA Logging
            if raw_term not in qa_log:
                qa_log[raw_term] = canonical_term
            elif qa_log[raw_term] != canonical_term:
                print(f"Warning: Inconsistent canonical mapping detected for raw term "
                      f"'{raw_term}'. Mapped to '{qa_log[raw_term]}' and now '{canonical_term}'.")

    # Join each cluster's keywords into a single tab-separated document
    for key, values in canonical_corpus.items():
        canonical_corpus[key] = "\t".join(values)
    canonical_corpus = dict(sorted(canonical_corpus.items()))

    # 5. Save the QA Log file
    qa_log_path = TFIDF_OUTPUT_DIR / "qa_canonical_keyword_mapping.json"
    print(f"\nSaving QA log to: {qa_log_path}")
    sorted_qa_log = dict(sorted(qa_log.items()))
    with open(qa_log_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_qa_log, f, indent=4, ensure_ascii=False)
    print(f"Total unique raw keywords processed and logged: {len(qa_log)}")

    # 5.1 Generate QA report: all unique canonical keywords
    all_keywords_sorted = sorted(all_canonical_keywords)
    qa_path = TFIDF_OUTPUT_DIR / "all_canonical_keywords_processed.txt"
    with open(qa_path, 'w', encoding='utf-8') as f:
        f.write(f"Total unique CANONICAL keywords processed: {len(all_keywords_sorted)}\n")
        f.write("=" * 80 + "\n\n")
        for kw in all_keywords_sorted:
            f.write(f"{kw.title()}\n")
    print(f"\nQA report saved to: {qa_path}")

    # 6. Apply TfidfVectorizer to the Canonical Corpus
    print("\nFitting TfidfVectorizer to the canonical corpus...")

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split('\t'),
        token_pattern=None,
        lowercase=False,
        norm=None,
    )

    X = vectorizer.fit_transform(canonical_corpus.values())

    # 6.1 Cluster-size normalization
    print("Applying cluster-size normalization...")
    cluster_ids_sorted = sorted(canonical_corpus.keys())
    paper_counts = np.array([cluster_paper_counts[cid] for cid in cluster_ids_sorted], dtype=float)

    for i, count in enumerate(paper_counts):
        if count > 0:
            X[i] = X[i] / count

    # 6.2 Correct IDF and apply final L2 normalization
    X = correct_tfidf(X, vectorizer, norm=NORM)

    # 7. Output Results
    print("\n--- Results ---")
    feature_names = vectorizer.get_feature_names_out()
    print(f"Total unique canonical keywords (features): {len(feature_names)}")

    if X.shape[0] > 0:
        cluster_vector = X[0].toarray().flatten()
        scores = pd.Series(cluster_vector, index=feature_names)
        scores = scores[scores > 0].sort_values(ascending=False)
        print(f"Highest TF-IDF scores (first cluster):\n{scores.head(10)}")
        print(f"\nSum of TF-IDF scores for the first cluster (L2 norm squared): {np.sum(cluster_vector**2):.4f}")

    return X, vectorizer, df


def sanitize_filename(name):
    """Sanitizes a string for use as a filename."""
    name = str(name).replace('\n', ' ')
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return re.sub(r'\s+', '_', name).strip()


def _aggregate_top_scores(X, vectorizer, cluster_ids, MODULARITY_META, top_n=3):
    """
    Helper function to aggregate the top N scores and their metadata for all clusters.
    """
    feature_names = vectorizer.get_feature_names_out()
    combined_scores = []

    for i, cluster_id in enumerate(cluster_ids):
        meta = MODULARITY_META.get(cluster_id, {})
        display_label = meta.get('label', f"Cluster {cluster_id}")
        plot_color = meta.get('color', '#1f77b4')

        cluster_vector = X[i].toarray().flatten()
        scores = pd.Series(cluster_vector, index=feature_names)
        scores = scores[scores > 0].sort_values(ascending=False)

        top_n_scores = scores.head(top_n)

        for rank, (keyword, score) in enumerate(top_n_scores.items()):
            combined_scores.append({
                'cluster_id': cluster_id,
                'cluster_label': display_label.replace('\n', ' '),
                'cluster_color': plot_color,
                'keyword': keyword.title(),
                'score': score,
                'rank': rank + 1
            })

    return combined_scores


def save_results(X, vectorizer, df, MODULARITY_META, top_n=20):
    """
    Calculates the TF-IDF scores for each cluster, saves the full results
    to a CSV, and generates a histogram for the top N keywords.
    """
    print("\n--- Generating TF-IDF Histograms and CSVs (with Meta Data) ---")

    feature_names = vectorizer.get_feature_names_out()

    df_cluster_ids = sorted(df[COMMUNITY_ATTR].unique())

    if len(df_cluster_ids) != X.shape[0]:
        print(f"Warning: Matrix rows ({X.shape[0]}) do not reliably match unique cluster IDs "
              f"({len(df_cluster_ids)}). Falling back to sequential numbering.")
        cluster_ids = list(range(X.shape[0]))
    else:
        cluster_ids = df_cluster_ids

    for i in range(X.shape[0]):
        cluster_id = cluster_ids[i]

        meta = MODULARITY_META.get(cluster_id, {})
        display_label = meta.get('label', f"Cluster {cluster_id}")
        plot_color = meta.get('color', '#1f77b4')

        cluster_vector = X[i].toarray().flatten()
        scores = pd.Series(cluster_vector, index=feature_names)
        scores = scores[scores > 0].sort_values(ascending=False)

        file_label = sanitize_filename(meta.get('label', f'{cluster_id}'))
        csv_path = TFIDF_OUTPUT_DIR / f"cluster_{cluster_id}_{file_label}_tfidf_scores.csv"

        df_output = pd.DataFrame({
            "canonical_keyword": scores.index,
            "tfidf_score": scores.values
        })
        df_output["canonical_keyword"] = df_output["canonical_keyword"].str.title()
        df_output.to_csv(csv_path, index=False)
        print(f"Saved TF-IDF scores CSV for {display_label} to: {csv_path.name}")

        top = df_output.head(top_n)

        if top.empty:
            print(f"No TF-IDF scores found for {display_label}, skipping plot.")
            continue

        labels = top['canonical_keyword'].tolist()[::-1]
        values = top['tfidf_score'].tolist()[::-1]

        plt.figure(figsize=(10, max(4, len(labels) * 0.35)))

        plt.barh(labels, values, color=plot_color)

        plt.xlabel('TF-IDF Score')
        plt.title(f"{display_label} - Top 20 Cluster-Distinguishing Keywords", loc='center')

        plt.tight_layout()

        png_path = TFIDF_OUTPUT_DIR / f"cluster_{cluster_id}_{file_label}_tfidf_histogram.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"Saved TF-IDF histogram for {display_label} to: {png_path.name}")

    return _aggregate_top_scores(X, vectorizer, cluster_ids, MODULARITY_META, top_n=3)


def plot_top_three_scores_combined(combined_scores_data, MODULARITY_META):
    """
    Plots the top 3 scores of each cluster with its respective color all in one plot.
    """
    if not combined_scores_data:
        print("No combined scores data to plot.")
        return

    df_combined = pd.DataFrame(combined_scores_data)

    df_combined['sort_label'] = df_combined['cluster_label'].str.split(':').str[-1].str.strip()
    df_combined = df_combined.sort_values(by=['sort_label', 'score'], ascending=[False, True])
    df_combined = df_combined.drop(columns=['sort_label'])

    df_combined['plot_label'] = df_combined.apply(
        lambda row: f"{row['keyword']} ({row['cluster_label']})", axis=1
    )

    labels = df_combined['plot_label'].tolist()
    values = df_combined['score'].tolist()
    colors = df_combined['cluster_color'].tolist()

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=meta['color'])
        for cluster_id, meta in sorted(MODULARITY_META.items())
        if cluster_id in df_combined['cluster_id'].unique()
    ]
    legend_labels = [
        meta['label'].replace('\n', ' ')
        for cluster_id, meta in sorted(MODULARITY_META.items())
        if cluster_id in df_combined['cluster_id'].unique()
    ]

    plt.figure(figsize=(12, max(6, len(labels) * 0.4)))
    plt.barh(labels, values, color=colors)

    plt.xlabel('TF-IDF Score (cluster-size normalized)')
    plt.title('Top 3 Canonical Keywords by Cluster', loc='center')

    plt.legend(legend_handles, legend_labels, title="Cluster", loc='lower right', framealpha=0.8)

    plt.tight_layout()

    png_path = TFIDF_OUTPUT_DIR / "combined_top_3_tfidf_histogram.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"\nSaved combined top 3 histogram: {png_path.name}")


# --- Main execution ---
synonym_dict_path = SYNONYMS_FILE
X_matrix, vectorizer_model, dataframe = calculate_canonical_tfidf(citation_network_df, synonym_dict_path)

top_three_scores = save_results(X_matrix, vectorizer_model, dataframe, MODULARITY_META)

plot_top_three_scores_combined(top_three_scores, MODULARITY_META)

print("Script execution successful for cluster")
