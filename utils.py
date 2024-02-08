import json
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def read_data(path):
    _, ext = os.path.splitext(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path, index_col=0)


def load_data(base_path):
    """Load data from disk.

    Args:
        base_path (str): Base path to data.

    Returns:
        pd.DataFrame: Data.
    """
    # Load data
    _, dataset_name = os.path.split(base_path)

    with open(os.path.join(base_path, f"{dataset_name}.meta.json")) as f:
        meta = json.load(f)

    format = meta["format"]

    train_data = read_data(os.path.join(base_path, f"{dataset_name}.{format}"))

    test_path = os.path.join(base_path, f"{dataset_name}_test.{format}")
    if os.path.exists(test_path):
        test_data = read_data(test_path)
    else:
        test_data = train_data

    return train_data, test_data, meta


def split_indices(base_path, repeats, test_sizes):
    train_idx_file = os.path.join(base_path, "train_indices.parquet")
    train_idx = pd.read_parquet(train_idx_file)

    test_idx_file = os.path.join(base_path, "test_indices.parquet")
    if os.path.exists(test_idx_file):
        test_idx = pd.read_parquet(test_idx_file)
    else:
        test_idx = slice(None)

    for i, (col_name, split) in enumerate(train_idx.items()):
        if i >= repeats:
            break

        if isinstance(test_idx, slice):
            test_split = test_idx
        else:
            test_split = test_idx[col_name].values

        for n in test_sizes:
            # Convert test size to number of samples if it is a fraction
            if n <= 1.0:
                n = n * len(split.values)
            n = int(n)
            n_f = n / len(split.values)

            info = {
                "i": i,
                "trainset_size": n,
                "trainset_fraction": n_f,
            }

            yield info, split.values[:n], test_split


def find_datasets(base_path: str, augmented=False) -> dict:
    """Find augmented datasets.

    Args:
        base_path (str): Base path to data.

    Returns:
        dict: Name, dataset mapping of augmented datasets.
    """
    datasets = {}
    for name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, name)):
            if augmented:
                filename = os.path.join(base_path, name, f"{name}_augmented.parquet")
            else:
                filename = os.path.join(base_path, name, f"{name}.parquet")
            if os.path.exists(filename):
                datasets[name] = filename
    return datasets


def find_augmented_datasets(base_path: str) -> dict:
    """Find augmented datasets.

    Args:
        base_path (str): Base path to data.

    Returns:
        dict: Name, dataset mapping of augmented datasets.
    """
    return find_datasets(base_path, augmented=True)


def fix_probabilities(true_classes, train_classes, pred_proba):
    if true_classes.shape[0] != pred_proba.shape[1]:
        print("Warning: not all classes are present in the train set")
        missing = set(true_classes).difference(set(train_classes))
        for m in sorted(missing):
            pred_proba = np.insert(pred_proba, m, 0, axis=1)
    return pred_proba


def score(y_true, y_pred, pred_proba):
    """Compute classification metrics.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.

    Returns:
        dict: Dictionary of metrics.
    """
    is_mutli_class = (len(pred_proba.shape) > 1) and (pred_proba.shape[1] > 2)
    average = "macro" if is_mutli_class else "binary"
    multi_class = "ovo" if is_mutli_class else "raise"

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
        "roc_auc": roc_auc_score(y_true, pred_proba, multi_class=multi_class),
        "r2": r2_score(y_true, y_pred),
    }


# Common options
dataset_option = click.option(
    "--dataset",
    "-d",
    type=click.Path(file_okay=False, exists=True),
    help="Path to the dataset folder",
    required=True,
)
repeats_option = click.option(
    "--repeats",
    "-r",
    type=int,
    default=20,
    help="Number of times the classification is repeated",
)
results_dir_option = click.option(
    "--results-dir", default="results/{dataset_name}", help="Path to the results folder"
)
models_dir_option = click.option(
    "--models-dir",
    default="models/{dataset_name}",
    help="Path to the folder to store models",
)
size_option = click.option(
    "--size",
    "-s",
    "test_sizes",
    multiple=True,
    default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    help="Sizes of the training set to consider",
)
