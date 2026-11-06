#!/usr/bin/env python3
"""Reproduce CDGAT-PUR AUROC and AUPR on processed CGGA-GLIOMA data."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cdgat_pur.data import load_graph_data, load_heldout_split, load_known_positive_genes
from cdgat_pur.training import set_seed, train_ensemble


METHOD = "CDGAT-PUR"


@dataclass
class TrainingConfig:
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    hidden_features: int = 16
    heads: int = 8
    dropout: float = 0.2
    negative_slope: float = 0.2
    lambda_init: float = -4.0
    pu_alpha: float = 1.0
    pu_pairs_per_positive: int = 1
    n_folds: int = 5
    max_epochs: int = 2000
    patience: int = 100
    current_seed: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def resolve_device(requested):
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    if requested == "cpu" or requested.startswith("cuda"):
        return torch.device(requested)
    raise ValueError("--device must be auto, cpu, cuda, or cuda:N.")


def summarize(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    critical = 2.776 if n == 5 else 1.96
    return {
        "mean": float(values.mean()),
        "sd": sd,
        "ci95": float(critical * sd / np.sqrt(n)) if n > 1 else 0.0,
        "n": n,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = TrainingConfig(max_epochs=args.max_epochs, patience=args.patience)

    print(f"Method: {METHOD}")
    print("Cohort: CGGA-GLIOMA")
    print("Omics: expression + methylation + mutation")
    print(f"Device: {device}")
    graph_data = load_graph_data(data_dir)
    genes = graph_data["genes"]
    known_positive = set(load_known_positive_genes(data_dir)).intersection(genes)
    print(f"Graph genes: {len(genes)}")
    print(f"Graph edges: {graph_data['edge_index'].shape[1]}")
    print(f"Known positives in graph: {len(known_positive)}")
    print(f"ANM expression samples: {graph_data['anm_sample_counts']}")

    metric_rows = []
    heldout_rows = []
    for seed in args.seeds:
        set_seed(seed)
        split = load_heldout_split(data_dir, seed, genes, known_positive)
        heldout = set(split["test_genes"])
        unlabeled = sorted(set(genes) - known_positive - heldout)
        mean_probability, _ = train_ensemble(
            graph_data,
            split["train_positive_genes"],
            unlabeled,
            device,
            config,
            seed,
        )

        mapping = graph_data["gene_to_index"]
        test_indices = np.asarray([mapping[gene] for gene in split["test_genes"]], dtype=np.int64)
        test_scores = mean_probability[test_indices]
        auroc = float(roc_auc_score(split["test_labels"], test_scores))
        aupr = float(average_precision_score(split["test_labels"], test_scores))
        metric_rows.append({"method": METHOD, "seed": seed, "auroc": auroc, "aupr": aupr})

        heldout_rows.extend(
            {
                "method": METHOD,
                "seed": seed,
                "Gene": gene,
                "Label": int(label),
                "score": float(score),
            }
            for gene, label, score in zip(
                split["test_genes"], split["test_labels"], test_scores
            )
        )
        print(f"seed={seed}: AUROC={auroc:.6f}, AUPR={aupr:.6f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(output_dir / "heldout_predictions.csv", index=False)
    summary_rows = []
    for metric in ["auroc", "aupr"]:
        summary_rows.append({"method": METHOD, "metric": metric, **summarize(metrics[metric])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
