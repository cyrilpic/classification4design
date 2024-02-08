import os
import time

import click
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, Normalizer
from tqdm.auto import tqdm

from utils import (
    fix_probabilities,
    load_data,
    split_indices,
    score,
    dataset_option,
    repeats_option,
    results_dir_option,
    size_option,
)


@click.command()
@dataset_option
@repeats_option
@results_dir_option
@size_option
def main(dataset, repeats, results_dir, test_sizes):
    dataset_name = dataset.split("/")[-1]
    if not os.path.exists(dataset):
        raise ValueError(f"Dataset {dataset} does not exist")

    results_path = results_dir.format(dataset_name=dataset_name)
    os.makedirs(results_path, exist_ok=True)

    results = []

    classifier = xgb.XGBClassifier()

    train_data, test_data, meta = load_data(dataset)

    x_columns = train_data.columns.drop(meta["label"])
    y_column = meta["label"]

    bar = tqdm(total=len(test_sizes) * repeats, desc=f"Training XGB ({dataset_name})")

    for info, train, test in split_indices(dataset, repeats, test_sizes):
        train_x = train_data.loc[train, x_columns]
        train_y = train_data.loc[train, y_column]
        test_x = test_data.loc[test, x_columns]
        test_y = test_data.loc[test, y_column]

        try:
            encoder = LabelEncoder()
            # normalizer = Normalizer()
            t0 = time.perf_counter()
            enc_train_y = encoder.fit_transform(train_y.values)
            # train_x = normalizer.fit_transform(train_x)
            classifier.fit(train_x, enc_train_y)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            # test_x = normalizer.transform(test_x)
            medi_pred = classifier.predict(test_x)
            medi_pred = encoder.inverse_transform(medi_pred)
            medi_test_time = time.perf_counter() - t0
            medi_proba = classifier.predict_proba(test_x)
        except Exception as e:
            print(e)
            raise e

        medi_proba = fix_probabilities(test_y.unique(), train_y.unique(), medi_proba)
        if medi_proba.shape[1] == 2:
            medi_proba = medi_proba[:, 1]

        medi_score = score(test_y.values, medi_pred, medi_proba)
        medi_score.update(
            {
                **info,
                "model": "xgb_default",
                "dataset": dataset_name,
                "train_time": train_time,
                "test_time": medi_test_time,
            }
        )

        results.append(medi_score)

        bar.update(1)

    bar.close()

    results = pd.DataFrame(results)
    results.to_parquet(os.path.join(results_path, "xgb_default.parquet"))


if __name__ == "__main__":
    main()
