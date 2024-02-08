import os
import time

import click
import pandas as pd
from autogluon.tabular import TabularPredictor
from tqdm.auto import tqdm

from utils import (
    dataset_option,
    fix_probabilities,
    load_data,
    models_dir_option,
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
@models_dir_option
@size_option
@click.option("--verbose", is_flag=True)
@click.option("--force", is_flag=True)
def main(
    dataset,
    repeats,
    results_dir,
    models_dir,
    test_sizes,
    verbose,
    force,
):
    verbosity = 2 if verbose else 0
    dataset_name = dataset.split("/")[-1]

    results_path = results_dir.format(dataset_name=dataset_name)
    os.makedirs(results_path, exist_ok=True)

    model_path = os.path.join(models_dir.format(dataset_name=dataset_name), "gluon")
    os.makedirs(model_path, exist_ok=True)

    train_data, test_data, meta = load_data(dataset)

    # x_columns = train_data.columns.drop(meta["label"])
    y_column = meta["label"]

    results = []

    bar = tqdm(
        total=len(test_sizes) * repeats, desc=f"Training AutoGluon ({dataset_name})"
    )

    for info, train, test in split_indices(dataset, repeats, test_sizes):
        train_df = train_data.loc[train]
        train_y = train_data.loc[train, y_column]
        test_df = test_data.loc[test]
        test_y = test_data.loc[test, y_column]

        n = info["trainset_fraction"]
        i = info["i"]

        medi_path = os.path.join(model_path, f"medium_{int(n*100):d}_{i}")
        best_path = os.path.join(model_path, f"best_{int(n*100):d}_{i}")

        if os.path.exists(medi_path) and not force:
            try:
                medi = TabularPredictor.load(medi_path)
                if medi._learner._time_fit_total is None:
                    medi = None
            except Exception:
                medi = None
        else:
            medi = None
        if medi is None:
            medi = TabularPredictor(
                label=y_column, path=medi_path, verbosity=verbosity
            ).fit(train_df, num_gpus=1, presets="medium_quality")

        if os.path.exists(best_path) and not force:
            try:
                best = TabularPredictor.load(best_path)
                if best._learner._time_fit_total is None:
                    best = None
            except Exception:
                best = None
        else:
            best = None
        if best is None:
            best = TabularPredictor(
                label=y_column, path=best_path, verbosity=verbosity
            ).fit(train_df, num_gpus=1, presets="best_quality")

        t0 = time.perf_counter()
        medi_pred = medi.predict(test_df)
        medi_test_time = time.perf_counter() - t0
        medi_proba = medi.predict_proba(test_df).values
        medi_proba = fix_probabilities(test_y.unique(), train_y.unique(), medi_proba)
        if medi_proba.shape[1] == 2:
            medi_proba = medi_proba[:, 1]

        t0 = time.perf_counter()
        best_pred = best.predict(test_df)
        best_test_time = time.perf_counter() - t0
        best_proba = best.predict_proba(test_df).values
        best_proba = fix_probabilities(test_y.unique(), train_y.unique(), best_proba)
        if best_proba.shape[1] == 2:
            best_proba = best_proba[:, 1]

        medi_score = score(test_y, medi_pred, medi_proba)
        medi_score.update(
            {
                **info,
                "model": "gluon_medium",
                "dataset": dataset_name,
                "train_time": medi._learner._time_fit_total,
                "test_time": medi_test_time,
            }
        )

        best_score = score(test_y, best_pred, best_proba)
        best_score.update(
            {
                **info,
                "model": "gluon_best",
                "dataset": dataset_name,
                "train_time": best._learner._time_fit_total,
                "test_time": best_test_time,
            }
        )

        results.append(medi_score)
        results.append(best_score)

        bar.update(1)

    bar.close()

    results = pd.DataFrame(results)
    results.to_parquet(os.path.join(results_path, "gluon.parquet"))


if __name__ == "__main__":
    main()
