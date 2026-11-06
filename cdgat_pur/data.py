"""Loading and validation of the processed CGGA-GLIOMA inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


GRAPH_DATABASES = [
    "kegg",
    "RegNetwork",
    "trrust",
    "JASPAR",
    "MOTIFMAP",
    "EVEX",
    "CHEA",
    "string",
    "CPDB",
    "IREF",
]


def _require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def _normalize_gene(value):
    if pd.isna(value):
        return None
    gene = str(value).strip().upper()
    return gene or None


def _standardize_rows(frame):
    values = frame.astype(float)
    means = values.mean(axis=1).to_numpy()[:, None]
    scales = values.std(axis=1).replace(0, np.nan).fillna(1.0).to_numpy()[:, None]
    return (values - means) / (scales + 1e-8)


def _load_omics(path, prefix, mutation=False):
    frame = pd.read_csv(_require(path))
    if "Gene" not in frame.columns:
        raise ValueError(f"{path} must contain a Gene column.")
    frame["Gene"] = frame["Gene"].map(_normalize_gene)
    frame = frame.dropna(subset=["Gene"]).groupby("Gene", as_index=True).mean(numeric_only=True)
    frame = frame.loc[frame.abs().sum(axis=1) != 0]
    if mutation:
        frame = frame.fillna(0.0).clip(lower=0, upper=1)
    else:
        frame = _standardize_rows(frame)
    frame.columns = [f"{prefix}_{column}" for column in frame.columns]
    return frame.astype(np.float32)


def _load_node_features(cohort_dir):
    cohort_dir = Path(cohort_dir)
    expression = _load_omics(cohort_dir / "expression.csv", "expr")
    methylation = _load_omics(cohort_dir / "methylation.csv", "meth")
    mutation = _load_omics(cohort_dir / "mutation.csv", "mut", mutation=True)
    features = pd.concat([expression, methylation, mutation], axis=1, join="inner").fillna(0.0)
    features = features.loc[features.abs().sum(axis=1) != 0]
    features.index.name = "Gene"
    return features.astype(np.float32)


def _load_prior_graph(graph_dir):
    graph_dir = Path(graph_dir)
    merged = None
    for database in GRAPH_DATABASES:
        path = _require(graph_dir / f"edges_{database}.csv")
        edges = pd.read_csv(path, usecols=["source", "target"]).dropna().drop_duplicates()
        edges["source"] = edges["source"].map(_normalize_gene)
        edges["target"] = edges["target"].map(_normalize_gene)
        edges = edges.dropna(subset=["source", "target"])
        edges = edges[edges["source"] != edges["target"]]
        edges[database] = 1.0
        merged = edges if merged is None else merged.merge(edges, on=["source", "target"], how="outer")
    return merged.fillna(0.0).drop_duplicates(["source", "target"])


def _load_anm_confidence(path, edges):
    columns = ["source", "target", "C_ANM", "n_samples", "anm_regressor", "anm_kappa"]
    anm = pd.read_csv(_require(path), usecols=columns)
    anm["source"] = anm["source"].map(_normalize_gene)
    anm["target"] = anm["target"].map(_normalize_gene)
    if len(anm) != len(edges):
        raise ValueError(f"ANM cache has {len(anm)} edges, but the filtered graph has {len(edges)}.")
    expected = edges[["source", "target"]].reset_index(drop=True)
    observed = anm[["source", "target"]].reset_index(drop=True)
    if not expected.equals(observed):
        mismatch = np.flatnonzero((expected != observed).any(axis=1).to_numpy())
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f"ANM cache edge order does not match the graph (first mismatch: {first}).")
    regressors = set(anm["anm_regressor"].dropna().astype(str).str.lower())
    kappas = set(pd.to_numeric(anm["anm_kappa"], errors="coerce").dropna().round(8))
    sample_counts = set(pd.to_numeric(anm["n_samples"], errors="coerce").dropna().astype(int))
    if regressors != {"krr"}:
        raise ValueError(f"Expected a KRR ANM cache, found: {sorted(regressors)}")
    if kappas != {1.0}:
        raise ValueError(f"Expected ANM kappa=1.0, found: {sorted(kappas)}")
    confidence = pd.to_numeric(anm["C_ANM"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    if np.any(confidence < -1.0) or np.any(confidence > 1.0):
        raise ValueError("C_ANM values must lie in [-1, 1].")
    return confidence, sorted(sample_counts)


def load_graph_data(data_dir):
    """Build the graph tensors using the same transformations as the study."""
    data_dir = Path(data_dir)
    cohort_dir = data_dir / "cgga_glioma"
    features = _load_node_features(cohort_dir)
    graph = _load_prior_graph(data_dir / "graphs")
    graph_genes = set(graph["source"]).union(graph["target"])
    overlap = sorted(set(features.index).intersection(graph_genes))
    graph = graph[graph["source"].isin(overlap) & graph["target"].isin(overlap)].copy()
    genes = sorted(set(graph["source"]).union(graph["target"]))
    features = features.loc[genes]
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    edge_pairs = graph[["source", "target"]].to_numpy()
    edge_index_np = np.vectorize(gene_to_index.__getitem__)(edge_pairs)
    edge_attr_np = graph[GRAPH_DATABASES].to_numpy(np.float32)
    confidence, sample_counts = _load_anm_confidence(
        cohort_dir / "anm_edge_confidence.csv", graph
    )

    degree = np.zeros((len(genes), 1), dtype=np.float32)
    nodes, counts = np.unique(edge_index_np, return_counts=True)
    degree[nodes, 0] = counts
    node_features = np.concatenate([features.to_numpy(np.float32), degree], axis=1)
    node_features = (
        node_features - node_features.mean(axis=0, keepdims=True)
    ) / (node_features.std(axis=0, keepdims=True) + 1e-8)

    return {
        "genes": np.asarray(genes),
        "gene_to_index": gene_to_index,
        "x": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index_np.T, dtype=torch.long).contiguous(),
        "edge_attr": torch.tensor(edge_attr_np, dtype=torch.float32),
        "causal_confidence": torch.tensor(confidence, dtype=torch.float32),
        "anm_sample_counts": sample_counts,
    }


def load_known_positive_genes(data_dir):
    path = _require(Path(data_dir) / "cgga_glioma" / "disease_genes.csv")
    frame = pd.read_csv(path)
    if not {"Gene", "Label"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain Gene and Label columns.")
    frame["Gene"] = frame["Gene"].map(_normalize_gene)
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
    return sorted(frame.loc[frame["Label"] == 1, "Gene"].dropna().unique())


def load_heldout_split(data_dir, seed, graph_genes, known_positive_genes):
    path = _require(Path(data_dir) / "splits" / f"heldout_split_seed{seed}.csv")
    frame = pd.read_csv(path)
    required = {"Gene", "Label", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}.")
    frame["Gene"] = frame["Gene"].map(_normalize_gene)
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower().str.strip()
    if frame["Gene"].duplicated().any():
        raise ValueError(f"{path} contains duplicated genes.")
    if set(frame["split"]) != {"train", "test"}:
        raise ValueError(f"{path} must contain train and test rows.")

    graph_gene_set = set(graph_genes)
    positive_set = set(known_positive_genes)
    missing = sorted(set(frame["Gene"]) - graph_gene_set)
    if missing:
        raise ValueError(f"Split genes absent from the graph: {missing[:10]}")
    labelled_positive = set(frame.loc[frame["Label"] == 1, "Gene"])
    if not labelled_positive.issubset(positive_set):
        raise ValueError("The split contains positive labels absent from disease_genes.csv.")
    if set(frame.loc[frame["Label"] == 0, "Gene"]) & positive_set:
        raise ValueError("A known positive gene is labelled as an unlabeled comparison gene.")

    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    test_labels = test["Label"].to_numpy(np.int64)
    if len(np.unique(test_labels)) != 2:
        raise ValueError("The held-out test set must contain both label classes.")
    return {
        "train_positive_genes": train.loc[train["Label"] == 1, "Gene"].tolist(),
        "test_genes": test["Gene"].tolist(),
        "test_labels": test_labels,
    }
