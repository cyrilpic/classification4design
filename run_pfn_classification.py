import os
import time

import click
import pandas as pd
from tabpfn import TabPFNClassifier
from tqdm.auto import tqdm

from utils import (
    dataset_option,
    fix_probabilities,
    load_data,
    repeats_option,
    results_dir_option,
    score,
    size_option,
    split_indices,
)


@click.command()
@dataset_option
@repeats_option
@results_dir_option
@size_option
@click.option("--custom-model", default=None)
@click.option("--model-name", default="pfn")
@click.option("--device", default="cuda:0")
def main(dataset, repeats, results_dir, test_sizes, custom_model, model_name, device):
    dataset_name = dataset.split("/")[-1]

    results_path = results_dir.format(dataset_name=dataset_name)
    os.makedirs(results_path, exist_ok=True)

    results = []

    if custom_model is not None:
        classifier = TabPFNClassifier(
            device=device, base_path="./", model_string=custom_model
        )
    else:
        classifier = TabPFNClassifier(device=device)

    train_data, test_data, meta = load_data(dataset)

    x_columns = train_data.columns.drop(meta["label"])
    y_column = meta["label"]

    bar = tqdm(total=len(test_sizes) * repeats, desc="Training PFN")

    for info, train, test in split_indices(dataset, repeats, test_sizes):
        train_x = train_data.loc[train, x_columns]
        train_y = train_data.loc[train, y_column]
        test_x = test_data.loc[test, x_columns]
        test_y = test_data.loc[test, y_column]

        t0 = time.perf_counter()
        classifier.fit(train_x, train_y.values, overwrite_warning=True)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        medi_pred = classifier.predict(test_x)
        medi_test_time = time.perf_counter() - t0
        medi_proba = classifier.predict_proba(test_x)

        medi_proba = fix_probabilities(test_y.unique(), train_y.unique(), medi_proba)
        if medi_proba.shape[1] == 2:
            medi_proba = medi_proba[:, 1]

        medi_score = score(test_y.values, medi_pred, medi_proba)
        medi_score.update(
            {
                **info,
                "model": model_name,
                "dataset": dataset_name,
                "train_time": train_time,
                "test_time": medi_test_time,
            }
        )

        results.append(medi_score)

        bar.update(1)

    bar.close()

    results = pd.DataFrame(results)
    results.to_parquet(os.path.join(results_path, f"{model_name}.parquet"))


if __name__ == "__main__":
    main()
