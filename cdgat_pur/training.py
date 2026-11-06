"""Five-fold positive-unlabeled ranking training for CDGAT-PUR."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm

from .model import CDGATPUR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pu_folds(positive_indices, unlabeled_indices, seed, n_folds=5):
    positive_indices = np.asarray(positive_indices, dtype=np.int64)
    unlabeled_indices = np.asarray(unlabeled_indices, dtype=np.int64)
    actual_folds = min(int(n_folds), len(positive_indices), len(unlabeled_indices))
    if actual_folds < 2:
        raise ValueError("At least two positive and two unlabeled training genes are required.")
    positive_splitter = KFold(actual_folds, shuffle=True, random_state=seed)
    unlabeled_splitter = KFold(actual_folds, shuffle=True, random_state=seed + 17)
    positive_folds = list(positive_splitter.split(positive_indices))
    unlabeled_folds = list(unlabeled_splitter.split(unlabeled_indices))
    folds = []
    for fold_id, ((p_train, p_val), (u_train, u_val)) in enumerate(
        zip(positive_folds, unlabeled_folds), start=1
    ):
        folds.append(
            {
                "fold_id": fold_id,
                "n_folds": actual_folds,
                "train_positive": positive_indices[p_train],
                "train_unlabeled": unlabeled_indices[u_train],
                "val_positive": positive_indices[p_val],
                "val_unlabeled": unlabeled_indices[u_val],
            }
        )
    return folds


def pu_ranking_loss(logits, positive_indices, unlabeled_indices, alpha, pairs_per_positive):
    positive_logits = logits[positive_indices].squeeze()
    positive_loss = F.softplus(-positive_logits).mean()
    positive_pairs = positive_indices.repeat_interleave(pairs_per_positive)
    choices = torch.randint(
        low=0,
        high=len(unlabeled_indices),
        size=(len(positive_pairs),),
        device=logits.device,
    )
    unlabeled_pairs = unlabeled_indices[choices]
    ranking_loss = F.softplus(
        -(logits[positive_pairs].squeeze() - logits[unlabeled_pairs].squeeze())
    ).mean()
    return positive_loss + alpha * ranking_loss


def _train_fold(graph_data, fold, device, config, fold_seed):
    set_seed(fold_seed)
    x = graph_data["x"].to(device)
    edge_index = graph_data["edge_index"].to(device)
    edge_attr = graph_data["edge_attr"].to(device)
    confidence = graph_data["causal_confidence"].to(device)
    train_positive = torch.tensor(fold["train_positive"], dtype=torch.long, device=device)
    train_unlabeled = torch.tensor(fold["train_unlabeled"], dtype=torch.long, device=device)
    val_positive = torch.tensor(fold["val_positive"], dtype=torch.long, device=device)
    val_unlabeled = torch.tensor(fold["val_unlabeled"], dtype=torch.long, device=device)

    model = CDGATPUR(
        in_features=x.shape[1],
        edge_features=edge_attr.shape[1],
        hidden_features=config.hidden_features,
        heads=config.heads,
        dropout=config.dropout,
        negative_slope=config.negative_slope,
        lambda_init=config.lambda_init,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_state = None
    wait = 0

    description = f"seed {config.current_seed}, fold {fold['fold_id']}/{fold['n_folds']}"
    for _ in tqdm(range(config.max_epochs), desc=description):
        model.train()
        logits = model(x, edge_index, edge_attr, confidence)
        loss = pu_ranking_loss(
            logits,
            train_positive,
            train_unlabeled,
            config.pu_alpha,
            config.pu_pairs_per_positive,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index, edge_attr, confidence)
            val_loss = pu_ranking_loss(
                val_logits,
                val_positive,
                val_unlabeled,
                config.pu_alpha,
                config.pu_pairs_per_positive,
            )
        value = float(val_loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x, edge_index, edge_attr, confidence)).cpu().numpy()
    return probabilities


def train_ensemble(graph_data, train_positive_genes, unlabeled_genes, device, config, seed):
    mapping = graph_data["gene_to_index"]
    positive_indices = np.asarray(
        [mapping[gene] for gene in sorted(set(train_positive_genes))], dtype=np.int64
    )
    unlabeled_indices = np.asarray(
        [mapping[gene] for gene in sorted(set(unlabeled_genes))], dtype=np.int64
    )
    folds = build_pu_folds(positive_indices, unlabeled_indices, seed + 1, config.n_folds)
    config.current_seed = seed
    predictions = []
    for fold in folds:
        fold_seed = seed + 100 + fold["fold_id"]
        predictions.append(_train_fold(graph_data, fold, device, config, fold_seed))
    matrix = np.vstack(predictions)
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
