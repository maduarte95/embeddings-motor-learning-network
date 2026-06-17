"""
Embedding loaders.

Each loader maps a list of document strings to a row-aligned ``(n, d)`` embedding
matrix, so the downstream pipeline (UMAP -> HDBSCAN -> c-TF-IDF, plus the
topic-quality metrics) can be run identically across embedding models. This is
the single source of truth for how each model turns text into vectors, shared by
``topic_modeling_new.py`` and the (forthcoming) embedding swap-test driver.

Design
------
- Loaders are **pure**: they take an already-built ``docs`` list and return
  embeddings in the same order. Corpus-change detection and on-disk caching are
  the caller's responsibility (see the fingerprint cache in
  ``topic_modeling_new.py``), which keeps one source of truth for the document
  set and lets a comparison driver layer its own per-model cache on top.
- **Input formatting is model-specific.** SPECTER2 expects title and abstract
  joined with ``[SEP]``; that separator is part of the ``docs`` the caller
  builds. Models added later should format their own input from a shared
  ``(title, abstract)`` pair so each model runs as designed, while the
  underlying corpus stays identical across models.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

SPECTER2_MODEL = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"


def embed_specter2(
    docs: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """SPECTER2 (base + proximity adapter) CLS-token embeddings.

    Parameters
    ----------
    docs
        Document strings, already formatted as ``title [SEP] abstract``.
    batch_size, max_length
        Tokenisation / batching parameters.
    device
        Torch device string; defaults to ``cuda`` if available else ``cpu``.

    Returns
    -------
    np.ndarray
        Shape ``(len(docs), 768)``, row-aligned with ``docs``.
    """
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model: {SPECTER2_MODEL}  (device: {device})")

    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_MODEL)
    model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL)
    model.load_adapter(
        SPECTER2_ADAPTER,
        source="hf",
        load_as="specter2",
        set_active=True,
    )
    # Fail loudly if the proximity adapter did not activate: without it we would
    # silently fall back to plain specter2-base, a substantially different (and
    # worse-suited) embedding space. The library may still log a benign
    # "none are activated" notice from unrelated base forward passes; this check
    # is the source of truth.
    active = model.active_adapters
    if active is None or "specter2" not in str(active):
        raise RuntimeError(
            f"SPECTER2 proximity adapter failed to activate (active_adapters={active!r}); "
            "embeddings would fall back to plain specter2_base."
        )
    model = model.to(device)
    model.eval()

    def embed_batch(texts: list[str]) -> np.ndarray:
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    print(f"  Embedding {len(docs):,} documents in batches of {batch_size}...")
    embeddings_list = []
    for i in tqdm(range(0, len(docs), batch_size), desc="  Embedding"):
        embeddings_list.append(embed_batch(docs[i:i + batch_size]))

    embeddings = np.vstack(embeddings_list)

    # Free GPU memory — the model is no longer needed past this point
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# ── Gemini (OpenRouter, OpenAI-compatible API) ───────────────────────────────

GEMINI_MODEL = "google/gemini-embedding-2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Symmetric task prefix (Google docs): the same corpus is used for clustering
# (BERTopic) and similarity (overlap), so one symmetric prefix is applied to
# every document. "clustering" matches the headline use; change here if you
# prefer "sentence similarity". (Do NOT use a search/retrieval prefix — that is
# asymmetric.)
GEMINI_TASK = "clustering"


def _gemini_input(doc: str, task: str) -> str:
    """Format a document for Gemini: drop the SPECTER-specific ``[SEP]`` join
    and apply Google's symmetric task prefix."""
    content = doc.replace(" [SEP] ", ". ").strip()
    return f"task: {task} | query: {content}"


GEMINI_CHECKPOINT = "data/embeddings/_gemini_checkpoint.npz"


def embed_gemini(
    docs: list[str],
    batch_size: int = 100,
    model: str = GEMINI_MODEL,
    task: str = GEMINI_TASK,
    max_retries: int = 5,
    checkpoint: str | None = None,
    save_every: int = 20,
) -> np.ndarray:
    """Gemini embeddings via OpenRouter's OpenAI-compatible endpoint.

    Requires ``OPENROUTER_API_KEY`` in the environment (loaded from ``.env``).
    Network + paid API: embeddings are cached by ``embedding_store`` so this runs
    once per (corpus, model). Returns ``(len(docs), d)`` float32, row-aligned.

    Robustness for this long, paid job:
    - validates each response (OpenRouter can return HTTP 200 with ``data=None``)
      and retries with exponential backoff;
    - checkpoints progress every ``save_every`` batches to ``checkpoint`` and
      resumes from it on re-run (matched to the exact corpus by fingerprint), so
      a mid-run failure does not throw away (or re-bill) completed work.
    """
    import hashlib
    import os
    import time
    from pathlib import Path

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set — copy EXAMPLE.env to .env and add your key."
        )
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    headers = {"X-Title": "motor-learning-embeddings"}

    inputs = [_gemini_input(d, task) for d in docs]
    fingerprint = hashlib.sha256("\x00".join(inputs).encode()).hexdigest()
    ckpt = Path(checkpoint or GEMINI_CHECKPOINT)

    # Resume only if the checkpoint was built for this exact corpus.
    vectors: list = []
    if ckpt.exists():
        saved = np.load(ckpt)
        if str(saved["fingerprint"]) == fingerprint and int(saved["n_done"]) <= len(inputs):
            vectors = list(saved["vectors"][: int(saved["n_done"])])
            print(f"  Resuming from checkpoint: {len(vectors):,}/{len(inputs):,} done")

    def _save() -> None:
        if not vectors:
            return
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp.npz")
        np.savez(tmp, vectors=np.asarray(vectors, dtype=np.float32),
                 n_done=len(vectors), fingerprint=fingerprint)
        tmp.replace(ckpt)  # atomic

    print(f"  Embedding {len(inputs):,} documents via {model} "
          f"(task={task}, batch={batch_size})...")

    starts = range(len(vectors), len(inputs), batch_size)
    for done_batches, i in enumerate(tqdm(starts, desc="  Embedding (gemini)"), 1):
        chunk = inputs[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    model=model, input=chunk,
                    encoding_format="float", extra_headers=headers,
                )
                if not resp.data or len(resp.data) != len(chunk):
                    raise RuntimeError(
                        f"unexpected response (data={getattr(resp, 'data', None)!r}, "
                        f"error={getattr(resp, 'error', None)!r})"
                    )
                break
            except Exception as e:  # transient null/rate-limit/network -> backoff
                if attempt == max_retries - 1:
                    _save()
                    raise RuntimeError(
                        f"gemini embedding failed at doc {i} after {max_retries} "
                        f"retries; progress saved to {ckpt} (re-run to resume). "
                        f"Last error: {e}"
                    ) from e
                wait = 2 ** attempt
                print(f"  retry {attempt + 1}/{max_retries} after error: {e} "
                      f"(waiting {wait}s)")
                time.sleep(wait)
        # API may return items out of order; sort by index to stay row-aligned.
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(item.embedding)
        if done_batches % save_every == 0:
            _save()

    arr = np.asarray(vectors, dtype=np.float32)
    if ckpt.exists():
        ckpt.unlink()  # success — discard the checkpoint
    return arr


# ── Registry ────────────────────────────────────────────────────────────────
# Maps an embedding key (also the storage key in embedding_store) to its loader
# and a human-readable model name. Selecting a space is a runtime choice (a
# --embedding flag), not a code edit — add a new model here and it becomes
# available to every driver. ``DEFAULT_EMBEDDING`` keeps existing commands
# working unchanged while remaining overridable.
DEFAULT_EMBEDDING = "specter2"

REGISTRY: dict[str, tuple] = {
    "specter2": (embed_specter2, "allenai/specter2_base+proximity"),
    "gemini": (embed_gemini, f"{GEMINI_MODEL} (task={GEMINI_TASK})"),
    # "scibert": (embed_scibert, "allenai/scibert_scivocab_uncased"),  # later
}


def get_loader(key: str) -> tuple:
    """Return ``(loader_fn, model_name)`` for embedding key ``key``."""
    if key not in REGISTRY:
        raise KeyError(
            f"unknown embedding {key!r}; known: {sorted(REGISTRY)}"
        )
    return REGISTRY[key]
