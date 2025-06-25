"""cluster_responses.py

Cluster and visualise "before" free-response answers contained in a sycophancy
results JSON file.

The script performs the following steps:
1. Loads *response_1* strings from every trial in the given JSON file (these are the
   answers produced *before* the reversal prompt).
2. Encodes each response with a SentenceTransformer (default: all-MiniLM-L6-v2).
3. Groups the embeddings with K-Means clustering (configurable cluster count).
4. Projects the high-dimensional embeddings to 2-D using UMAP (falls back to PCA
   if *umap-learn* is not installed).
5. Generates an interactive Plotly scatter plot (points coloured by cluster,
   tool-tip shows the full response text) and saves it as an HTML file next to
   the input JSON (e.g. *..._clusters.html*).

Usage
-----
python cluster_responses.py path/to/free_response_*.json [--clusters 10] [--model all-MiniLM-L6-v2]

Dependencies
------------
 pip install sentence-transformers scikit-learn plotly umap-learn pandas tqdm

Note: UMAP gives a more structure-preserving layout, but if *umap-learn* is not
available the script will transparently fall back to PCA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import threadpoolctl

# ---------------------------------------------------------------------------
# Patch threadpoolctl (& sklearn alias) early to avoid Windows MKL bug
# ---------------------------------------------------------------------------

if not hasattr(threadpoolctl, "_orig_threadpool_info"):

    threadpoolctl._orig_threadpool_info = threadpoolctl.threadpool_info  # type: ignore[attr-defined]

    def _safe_threadpool_info(*args, **kwargs):  # type: ignore[override]
        try:
            return threadpoolctl._orig_threadpool_info(*args, **kwargs)  # type: ignore[attr-defined]
        except AttributeError:
            return []

    threadpoolctl.threadpool_info = _safe_threadpool_info  # type: ignore[assignment]

    # Patch sklearn's alias as well (if sklearn already imported)
    try:
        import sklearn.utils.fixes as _sk_fixes  # type: ignore

        _sk_fixes.threadpool_info = threadpoolctl.threadpool_info  # type: ignore[attr-defined]
    except ImportError:
        # sklearn not yet imported; when it is, it'll pick up patched function
        pass


try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    sys.exit("❌ sentence-transformers is required. `pip install sentence-transformers`. ")

# Attempt to import UMAP for nicer visualisation; fall back to PCA if missing.
try:
    import umap  # type: ignore

    def reduce_dim(embeddings: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        return reducer.fit_transform(embeddings)

except ImportError:  # pragma: no cover

    def reduce_dim(embeddings: np.ndarray) -> np.ndarray:  # type: ignore[override]
        print("Warning: umap-learn not found, falling back to PCA for 2-D projection.")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster 'before' responses and visualise them.")
    parser.add_argument("json_path", type=Path, help="Path to free_response_*.json file")
    parser.add_argument("--clusters", "-k", type=int, default=5, help="Number of clusters for K-Means (default: 10)")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="If provided, only include records whose question_type exactly matches this value (e.g. 'breakup').",
    )
    return parser.parse_args()


def load_before_responses(path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Return list of *response_1* strings and corresponding IDs for hover text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: List[str] = []
    ids: List[str] = []
    types: List[str] = []

    for rec in data.get("records", []):
        qid = rec.get("question_id", "?")
        for trial in rec.get("trials", []):
            run_idx = trial.get("run_idx", "?")
            resp = trial.get("response_1")
            if resp:
                texts.append(resp)
                ids.append(f"{qid}-run{run_idx}")
                types.append(rec.get("question_type", "unknown"))
    if not texts:
        raise ValueError("No response_1 entries found in JSON – cannot cluster.")
    return texts, ids, types


def main() -> None:
    args = parse_args()

    if not args.json_path.exists():
        sys.exit(f"❌ File not found: {args.json_path}")

    print("Loading responses ...")
    texts, ids, qtypes = load_before_responses(args.json_path)
    if args.task_type:
        mask = [qt == args.task_type for qt in qtypes]
        if not any(mask):
            sys.exit(f"No records found with question_type = '{args.task_type}'. Available types: {set(qtypes)}")
        texts = [t for t, m in zip(texts, mask) if m]
        ids = [i for i, m in zip(ids, mask) if m]
        qtypes = [qt for qt, m in zip(qtypes, mask) if m]
        print(f"Filtered to {len(texts)} responses of type '{args.task_type}'.")
    else:
        print(f"Loaded {len(texts)} responses.")

    print("Encoding with SentenceTransformer ... (this may take a while)")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("Dimensionality reduction ...")
    embeddings_2d = reduce_dim(embeddings)

    print("Clustering with AgglomerativeClustering ...")
    clusterer = AgglomerativeClustering(n_clusters=args.clusters)
    labels = clusterer.fit_predict(embeddings)

    # Plot with Plotly
    print("Building interactive scatter plot ...")
    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover
        sys.exit("❌ plotly is required. `pip install plotly`. ")

    df_data = {
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "type": qtypes,
        "cluster": labels.astype(str),
        "id": ids,
        "text": texts,
    }
    fig = px.scatter(
        df_data,
        x="x",
        y="y",
        color="type",
        symbol="cluster",
        hover_data={"id": True, "text": True, "type": True, "cluster": True, "x": False, "y": False},
        title=f"Sentence-Transformer Clusters (k={args.clusters}) – {args.json_path.name}",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))

    out_html = args.json_path.parent / f"{args.json_path.stem}_clusters.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Cluster plot saved to {out_html}")


if __name__ == "__main__":
    main() 