"""
Plots for Jaccard overlap results produced by run_overlap.py.

Three figures:
  1. headline.png      — mean Jaccard per citation method, SEM error bars,
                         dotted line at null mean.
  2. by_topic.png      — top-N and bottom-N BERTopic topics for one method
                         (default: 'combined'), filtered by minimum size.
  3. by_community.png  — same for Leiden communities at the chosen resolution.

Error bars are 1 SEM (= std / sqrt(n)).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("data/overlap_results")
FIG_DIR = RESULTS_DIR / "figures"
TOPIC_WORDS = Path("data/topic_words_new.csv")
TFIDF_DIR = Path("tf_idf_results")

METHODS = ["bibliographic_coupling", "co_citation", "combined"]
METHOD_COLORS = {
    "bibliographic_coupling": "#4C72B0",
    "co_citation": "#DD8452",
    "combined": "#55A868",
}
HIGH_COLOR = "#55A868"
LOW_COLOR = "#C44E52"


def _topic_label_lookup(path: Path = TOPIC_WORDS) -> dict[int, str]:
    if not path.exists():
        return {}
    tw = pd.read_csv(path)
    out: dict[int, str] = {}
    for _, row in tw.iterrows():
        words = str(row["words"]).split(" | ")[:3]
        out[int(row["topic_id"])] = ", ".join(words)
    return out


def _community_label_lookup(tfidf_dir: Path = TFIDF_DIR, top_k: int = 3) -> dict[int, str]:
    """Map community id -> top-k TF-IDF keywords, from keywords_analysis.py output.

    Reads the per-community ``cluster_<id>_..._tfidf_scores.csv`` files (one row
    per keyword, sorted by descending TF-IDF). Communities that keywords_analysis
    did not process (beyond its TOP_N_COMMUNITIES) simply won't appear here, and
    the plot falls back to a bare ``comm <id>`` label for them.
    """
    out: dict[int, str] = {}
    if not tfidf_dir.exists():
        return out
    for path in sorted(tfidf_dir.glob("cluster_*_tfidf_scores.csv")):
        m = re.match(r"cluster_(\d+)_", path.name)
        if not m:
            continue
        try:
            kw = pd.read_csv(path)["canonical_keyword"].head(top_k).astype(str).tolist()
        except (OSError, KeyError, pd.errors.ParserError):
            continue
        if kw:
            out[int(m.group(1))] = ", ".join(kw)
    return out


def _null_mean(method: str, results_dir: Path = RESULTS_DIR) -> float:
    summary = pd.read_csv(results_dir / "summary.csv")
    return float(summary.loc[summary["method"] == method, "null_mean"].iloc[0])


def plot_headline(results_dir: Path = RESULTS_DIR, out_dir: Path = FIG_DIR) -> Path:
    """Mean Jaccard per method with SEM error bars + null mean line."""
    means: list[float] = []
    sems: list[float] = []
    ns: list[int] = []
    for method in METHODS:
        pp = pd.read_csv(results_dir / f"{method}_per_paper.csv")
        means.append(float(pp["jaccard"].mean()))
        sems.append(float(pp["jaccard"].sem()))
        ns.append(len(pp))

    null_mean = pd.read_csv(results_dir / "summary.csv")["null_mean"].mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(METHODS))
    ax.bar(
        x, means, yerr=sems, capsize=8,
        color=[METHOD_COLORS[m] for m in METHODS],
        edgecolor="black", linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in METHODS])
    ax.axhline(null_mean, ls="--", color="gray", lw=1.2, label=f"null mean = {null_mean:.4f}")
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + sems[i] + 0.002, f"{m:.3f}\n(n={n:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean Jaccard")
    ax.set_title("Embedding-NN vs citation-NN overlap (k=10)\nerror bars: ± 1 SEM")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(0, max(means) * 1.35)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "headline.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_by_topic(
    method: str = "combined",
    n_show: int = 15,
    min_size: int = 10,
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = FIG_DIR,
) -> Path:
    """Top-N and bottom-N topics by mean Jaccard, side-by-side."""
    agg = pd.read_csv(results_dir / f"{method}_by_topic.csv")
    agg = agg[(agg["topic"] >= 0) & (agg["n"] >= min_size)].copy()
    agg = agg.sort_values("mean", ascending=False).reset_index(drop=True)

    top = agg.head(n_show).copy()
    bottom = agg.tail(n_show).copy().iloc[::-1].reset_index(drop=True)  # worst on top

    labels = _topic_label_lookup()
    null_mean = _null_mean(method, results_dir)

    def make_label(t: int, n: int) -> str:
        return f"{int(t):>3d} · {labels.get(int(t), '')[:38]} (n={int(n)})"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
    for ax, df, title, color in [
        (axes[0], top, f"Top {n_show} topics", HIGH_COLOR),
        (axes[1], bottom, f"Bottom {n_show} topics", LOW_COLOR),
    ]:
        sem = df["std"] / np.sqrt(df["n"])
        y = np.arange(len(df))
        ax.barh(y, df["mean"], xerr=sem, capsize=3,
                color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([make_label(t, n) for t, n in zip(df["topic"], df["n"])], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(null_mean, ls="--", color="gray", lw=1)
        ax.set_xlabel("Mean Jaccard")
        ax.set_title(title)

    fig.suptitle(
        f"Per-topic overlap — {method}, k=10, min size = {min_size}\n"
        f"dotted line: null mean ({null_mean:.4f});  error bars: ± 1 SEM"
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"by_topic_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_community(
    method: str = "combined",
    n_show: int = 15,
    min_size: int = 30,  # matches keywords_analysis.MIN_COMMUNITY_SIZE so every bar has a label
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = FIG_DIR,
) -> Path:
    """Top-N and bottom-N citation communities by mean Jaccard."""
    agg = pd.read_csv(results_dir / f"{method}_by_community.csv")
    agg = agg[agg["n"] >= min_size].copy()
    agg = agg.sort_values("mean", ascending=False).reset_index(drop=True)

    top = agg.head(n_show).copy()
    bottom = agg.tail(n_show).copy().iloc[::-1].reset_index(drop=True)
    null_mean = _null_mean(method, results_dir)

    labels = _community_label_lookup()

    def make_label(c, n: int) -> str:
        # community column may be string-cast floats; coerce to int when possible
        try:
            cid = int(float(c))
        except (ValueError, TypeError):
            cid = c
        kw = labels.get(cid, "") if isinstance(cid, int) else ""
        if kw:
            return f"{cid:>3d} · {kw[:38]} (n={int(n)})"
        return f"comm {cid} (n={int(n)})"

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    for ax, df, title, color in [
        (axes[0], top, f"Top {n_show} communities", HIGH_COLOR),
        (axes[1], bottom, f"Bottom {n_show} communities", LOW_COLOR),
    ]:
        sem = df["std"] / np.sqrt(df["n"])
        y = np.arange(len(df))
        ax.barh(y, df["mean"], xerr=sem, capsize=3,
                color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([make_label(c, n) for c, n in zip(df["community"], df["n"])], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(null_mean, ls="--", color="gray", lw=1)
        ax.set_xlabel("Mean Jaccard")
        ax.set_title(title)

    fig.suptitle(
        f"Per-community overlap — {method}, k=10, min size = {min_size}\n"
        f"dotted line: null mean ({null_mean:.4f});  error bars: ± 1 SEM"
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"by_community_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p1 = plot_headline()
    p2 = plot_by_topic()
    p3 = plot_by_community()
    for p in (p1, p2, p3):
        print(f"Saved {p}")
