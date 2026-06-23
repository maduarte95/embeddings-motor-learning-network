# LLM Embeddings for the Motor Learning Citation Network

1. The file data/citation_network_selected.graphml corresponds to the result of motor-learning-network, but filtered so nodes with less than 5 edges are excluded.

2. The network is available online [here](https://alfredohernandezinostroza.github.io/citation-network-ncm/).

3. The topics were obtained by running `topic_modeling.py`. It's an implementation of BERTopic with Specter2.

4. A visualization of the discovered topics over the network is available [here](https://alfredohernandezinostroza.github.io/topics-ncm/)

5. The script did not save the embeddings. The upgraded version of the script that does save the embeddings, among other things, is `topic_modeling_new.py`. Note that the result is not the online version

6. The main other feature of this script is that it will not recalculate the embeddings if the input graph is the same.

# Branch comparison_metrics

1. Topics are obtained by `topic_modeling_new.py`. `topic_modeling.py` is deprecated.

2. Includes helper scripts for generating data for visualizations in `build_semantic_web_data.py` (for semantic space visualization) and `build_web_data.py` (for topic/citation community visualization). Code for the visualization of 2D projection of the embedding space is available [in this repo](https://github.com/maduarte95/motor-semantic-viz).

3. Nearest neighbors overlap metrics between semantic and citation spaces can be calculated with `run_overlap.py`. Plots can be generated with `plot_overlap.py`. Metrics and plots will be stored in `data/overlap_results`.

4. A data exploration notebook is provided in `inspect_papers.ipynb`, including a simple paper recommendation based on overlap metrics.

5.  Embeddings are calculated separately from topic modelling and swappable via command line arguments (see "How to use").

    Supported embeddings:

        - SPECTER2 (default)
        - Gemini embedding 2 (requires OpenRouter API key)

6. Topic quality is evaluated with `topic_quality.py`: NPMI coherence and topic diversity (higher = better), outlier/coverage %, and an author-overlap-within-topics diagnostic (vs a label-shuffle null). Held under a fixed pipeline, these also compare embedding models (the "swap test"). Coherence is computed directly from document co-occurrence (no gensim dependency).

7. A recommender (`community_recommender.py`) surfaces, within each semantic topic, citation-community pairs that are semantically together but far apart in citation space — same problem space, no citation cross-talk. Targets a topic, paper, or author, with a swappable citation-space connectedness measure (see "How to use").


To be implemented:
- characterization of the embedding space (e.g.: anisotropy, intrinsic dimensionality, hubness, 2D trustworthiness, boostrapped neighborhood stability)
- support for SentenceTransformers embeddings (e.g. Stella400M/1.5B, F2LLM-4B, Qwen3-Embedding-8B)
- representation similarity analyses (RSA, Procrustes)


## How to use

```bash
pixi install
```

### Pipeline — embeddings & topics
```bash
# Compute embeddings + fit BERTopic, write topics back to the graph.
# Uses the default embedding model (specter2); caches embeddings, the BERTopic
# model, and a graph fingerprint so reruns are fast.
pixi run python topic_modeling_new.py

# Force a full recompute (ignore all caches)
pixi run python topic_modeling_new.py --recompute

# Run with a different embedding model (must be registered in
# embedding_loaders.REGISTRY); output goes to data/embeddings/<key>.npz
pixi run python topic_modeling_new.py --embedding gemini
```

### Overlap analysis (embedding-NN vs citation-NN)
```bash
# Per-paper / per-topic / per-community Jaccard overlap + null baseline.
# Results written to data/overlap_results/
pixi run python run_overlap.py                 # default: specter2
pixi run python run_overlap.py --embedding gemini

# Plots from the overlap results (files in data/overlap_results)
pixi run python plot_overlap.py                # topic labels: specter2
pixi run python plot_overlap.py --embedding gemini
```

> ⚠️ **Note:** `plot_overlap.py --embedding <key>` only selects which model's
> topic-word *labels* are shown. The overlap results in `data/overlap_results/`
> are **not** keyed by model — `run_overlap.py` overwrites them each run — so the
> plotted numbers reflect whichever model ran `run_overlap.py` last. Always pass
> the same `--embedding` to `plot_overlap.py` that you used for `run_overlap.py`,
> or the labels won't match the data.

### Topic quality
```bash
# NPMI coherence + diversity + outlier % + author-overlap diagnostic
# for a model's saved topics.
pixi run python topic_quality.py                  # default: specter2
pixi run python topic_quality.py --embedding gemini
```

### Topic granularity sweep
```bash
# Sweep HDBSCAN min_cluster_size (UMAP fixed) and tabulate
# n_topics / coverage / npmi / diversity to pick a granularity.
# Output: data/topic_sweep_<key>.csv
pixi run python run_topic_sweep.py                                   # default grid
pixi run python run_topic_sweep.py --min-cluster-sizes 10,20,40,80
pixi run python run_topic_sweep.py --embedding gemini
```

> ⚠️ **Note:** the sweep **re-runs HDBSCAN (and c-TF-IDF) from scratch** for every
> grid point — it does *not* use the cached BERTopic model, and the corpus
> fingerprint does not bust that cache on a parameter change. The 5-D clustering
> UMAP is computed once and cached (`data/umap_5d_<key>.npz`; `--recompute-umap`
> to rebuild). Expect a minute or two per run depending on the grid size.

### Recommender
```bash
# Within each topic, surface citation-community pairs that are semantically
# together but far apart in citation space. Output:
# data/recommender_results/recommendations_<embedding>_<method>.csv
pixi run python community_recommender.py --embedding gemini --method conductance --all

# Single targets (mutually exclusive)
pixi run python community_recommender.py --embedding gemini --topic 5
pixi run python community_recommender.py --embedding gemini --paper n123
pixi run python community_recommender.py --embedding gemini --author "rizzolatti"
```

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--embedding` | `specter2`, `gemini` | `specter2` | Which topic model is the semantic ground truth |
| `--method` | `bibliographic_coupling`, `co_citation`, `combined`, `conductance` | `bibliographic_coupling` | Citation-space connectedness measure |
| `--topic N` / `--paper nID` / `--author "name"` / `--all` | — | — | Target (mutually exclusive) |
| `--k` | int (`0` = all pairs) | `10` | Bottom-k least-connected pairs |
| `--min-papers` | int | `10` | Min papers a community needs *in a topic* to qualify |

> ⚠️ **Note:** `conductance` carries a
> partner-size baseline problem (big, prolific communities rank as "disconnected"
> regardless), so when comparing embeddings check with the other metrics (e.g. `combined`).

### Recommender explorer (UI)
```bash
# Interactive Streamlit app over community_recommender.py: search by topic,
# paper, or author and browse the least-connected community pairs, then drill
# into the papers on each side. Embedding / method / k / min-papers are widgets;
# a sidebar panel explains how each connectedness measure is computed.
pixi run streamlit run recommender_app.py

# In a browser, navigate to:
http://localhost:8501/
```

Search modes (top of the page):
- **Topic** (dropdown labelled with topic
keywords)
- **Paper** (search by title or node id)
- **Author** (name substring)
- **All topics** — pools every topic's candidate pairs into one table and ranks them globally (top-N most disconnected across the corpus, with scores always
**intra-topic**). in every mode (each pair is scored only against its own topic's
communities);

Community ids are shown with the labels
from `data/community_names.json`.


### Visualization assets
```bash
# Semantic 2D-UMAP web assets (uses a chosen embedding space)
pixi run python build_semantic_web_data.py                  # default: specter2
pixi run python build_semantic_web_data.py --embedding gemini
pixi run python build_semantic_web_data.py --recompute      # ignore UMAP cache


# Citation-graph web assets
pixi run python build_web_data.py
```

Semantic web assets are written **per embedding model** so several can coexist
and be toggled in the frontend:

```
web_semantic/data/
  edges_out.bin / edges_in.bin   (shared; citations, model-independent)
  abstracts.json                 (shared; text, model-independent)
  <key>/
    nodes.json                   (2D-UMAP layout + per-node topic/cluster)
    topics.json                  (topic names/colors/centroids)
    communities.json             (community centroids — in this model's UMAP space)
```

Run `build_semantic_web_data.py --embedding <key>` once per model; the frontend
picks a `<key>` and loads that subdir plus the shared top-level files.

> ⚠️ **Note:** the shared graphml keeps citations + Leiden cluster (model-independent);
> its `topic` node attribute reflects only the most recent topic run. In `build_semantic_web_data.py`
> per-node topic read from document_topics_{key}.csv (not the graphml topic attr!) so the stored topic
> information matches the embeddings. `build_web_data.py` might still use only the last run topics.

### Interactive inspection
```bash
# Poke at individual papers, topics, communities, and NN lists.
# Set EMBEDDING_KEY in the setup cell to switch embedding space.
pixi run jupyter lab inspect_papers.ipynb
```

### Other utilities
```bash
pixi run python calculate_conductance.py    # conductance per Leiden community
pixi run python keywords_analysis.py        # keyword labels for citation communities
pixi run python compare_partitions.py       # compare citation/topic partitions
pixi run python graphml_to_parquet.py       # convert graphml -> parquet
```

### Embedding spaces
Embeddings are stored per-model and self-describing at `data/embeddings/<key>.npz`
(vectors + node_ids + metadata). To add a new model, register a loader in
`embedding_loaders.REGISTRY`, then pass `--embedding <key>` to any script above.
The default is `specter2` (`embedding_loaders.DEFAULT_EMBEDDING`).

### Setting up OpenRouter API key (required for Gemini embeddings)
1. Copy `EXAMPLE.env` to `.env`
2. Replace placeholder `your_openrouter_api_key_here` with OpenRouter API key

`google/gemini-embedding-2` returns 3072-dim vectors and is applied with Google's
symmetric task prefix `task: clustering | query: {content}` (set in
`embedding_loaders.GEMINI_TASK`). The first run is a **paid API call over the
whole corpus** (~14.5k documents, several hundred batched requests with retry
backoff) — it is cached in `data/embeddings/gemini.npz`, so subsequent steps just
load it. Generate it with:

```bash
pixi run python topic_modeling_new.py --embedding gemini
```


## Metrics explanation

### Overlap

*Produced by `run_overlap.py` (`embedding_neighbors.py` + `citation_neighbors.py` + `jaccard_overlap.py`); written to `data/overlap_results/`; plotted by `plot_overlap.py`.*

How much do a paper's **nearest neighbors in embedding space** agree with its
**nearest neighbors in citation space**? For each paper we take its top-_k_
(k=10) neighbors in each space and compare the two sets.

- **Embedding NN** — top-k by cosine similarity of the embedding vectors.
- **Citation NN** — top-k by one of three classic bibliometric similarities:
  - `bibliographic_coupling` — cosine of out-citation (reference) vectors: papers
    are similar if they **cite the same references** (Kessler 1963).
  - `co_citation` — cosine of in-citation vectors: papers are similar if they are
    **cited together** by the same later papers (Small 1973).
  - `combined` — the average of the two.
- **Jaccard index** — per paper, `|emb_NN ∩ cit_NN| / |emb_NN ∪ cit_NN|`; 0 = the
  two spaces disagree completely, 1 = identical neighbor sets. Aggregated as a
  mean/median overall, per BERTopic topic, and per Leiden community.
- **Null baseline & z-score** — citation-NN lists are randomly reassigned across
  papers (preserving list sizes) to get the Jaccard expected by chance; the
  z-score `(real − null_mean) / null_std` says how far above chance the real
  overlap sits.

Note: SPECTER2 is citation-trained, so high overlap is expected.

### Topic quality

*Produced by `topic_quality.py` (metric functions reusable via `evaluate_topics(key)`).*

Whether the discovered topics are good, and (under a fixed pipeline) which
embedding produces better topics. Interpret metrics in conjunction with each other.

- **NPMI coherence** — do a topic's top-10 words actually **co-occur** in the
  abstracts? Normalised pointwise mutual information per word pair, averaged over
  pairs and topics (Bouma 2009; Lau et al. 2014). **higher = more
  interpretable**. Computed directly from document co-occurrence (no gensim).
- **Topic diversity** — fraction of **unique words** across all topics' top-10
  (Dieng et al. 2020); **higher = less redundant** topics. Low values flag many
  topics repeating the same generic words.
- **Outliers / coverage** — share of papers in the `-1` (HDBSCAN noise) bucket;
  coverage = 1 − outliers.
- **Author overlap within topics** *(diagnostic)* — mean number of **shared
  authors per within-topic paper pair**, vs a label-shuffle null (z-score) i.e. how much topics coincide with author groups. Should be interpreted as a noisy signal of topic quality (a good semantic topic could have a broad author coverage).
  Might under-count spelling variants.

### Topic granularity sweep

*Produced by `run_topic_sweep.py`; written to `data/topic_sweep_{key}.csv`.*

`run_topic_sweep.py` varies HDBSCAN's `min_cluster_size` (topic count is
emergent, not set directly) with everything upstream fixed, tabulating
`n_topics`, `coverage`, `npmi`, and `diversity` per setting. Pick a granularity where coherence/diversity are near their peak at acceptable coverage.