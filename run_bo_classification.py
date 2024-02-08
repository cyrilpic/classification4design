import os
import time

import click
import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.parallel import Parallel, delayed
from smac import HyperparameterOptimizationFacade, Scenario
from tqdm.auto import tqdm

from tabular.classifiers import decision_trees, knn, mlp, svm, xgb
from utils import (
    fix_probabilities,
    load_data,
    score,
    split_indices,
    dataset_option,
    repeats_option,
    results_dir_option,
    size_option,
    models_dir_option,
)


@click.command()
@dataset_option
@repeats_option
@results_dir_option
@models_dir_option
@size_option
@click.option("--model-type", default="xgb")
@click.option("--n-trials", "-n", default=100)
@click.option("--n-jobs", default=-1)
@click.option("--n-workers", default=1)
def main(
    dataset,
    repeats,
    results_dir,
    models_dir,
    test_sizes,
    model_type,
    n_trials,
    n_jobs,
    n_workers,
):
    dataset_name = dataset.split("/")[-1]

    if not os.path.exists(dataset):
        raise ValueError(f"Dataset {dataset} does not exist")

    results_path = results_dir.format(dataset_name=dataset_name)
    os.makedirs(results_path, exist_ok=True)

    bo_name = "bo" if n_trials == 100 else f"bo_{n_trials}"

    model_path = os.path.join(
        models_dir.format(dataset_name=dataset_name), bo_name, model_type
    )
    os.makedirs(model_path, exist_ok=True)

    results = []

    model_types = {
        "decision_trees": decision_trees.DecisionTree,
        "knn": knn.KNN,
        "xgb": xgb.XGB,
        "svm": svm.SVM,
        "mlp": mlp.FastAI,
    }

    model = model_types[model_type]

    train_data, test_data, meta = load_data(dataset)

    x_columns = train_data.columns.drop(meta["label"])
    y_column = meta["label"]

    bar = tqdm(
        total=len(test_sizes) * repeats,
        desc=f"Training BO {model_type} ({dataset_name})",
    )

    for info, train_i, test_i in split_indices(dataset, repeats, test_sizes):
        n = info["trainset_fraction"]
        i = info["i"]

        train_x = train_data.loc[train_i, x_columns]
        train_y = train_data.loc[train_i, y_column]
        test_x = test_data.loc[test_i, x_columns]
        test_y = test_data.loc[test_i, y_column]

        t0 = time.perf_counter()
        encoder = LabelEncoder()
        scaler = StandardScaler()

        train_y[:] = encoder.fit_transform(train_y)
        train_x[:] = scaler.fit_transform(train_x)

        def train(
            config: Configuration,
            seed: int = 0,
        ) -> float:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            parallel = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs", timeout=60)
            scores = parallel(
                delayed(model.fit_score)(config, train_x, train_y, train, test, seed)
                for train, test in cv.split(train_x, train_y)
            )
            return 1 - np.mean(scores)

        configspace = model.parameters(
            num_classes=meta["num_classes"], n_samples=train_x.shape[0]
        )

        run_name = f"run_{int(n*100):d}_{i}"

        scenario = Scenario(
            configspace,
            name=run_name,
            deterministic=True,
            n_trials=n_trials,
            output_directory=model_path,
            n_workers=n_workers,
            seed=12345,
        )

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(
            scenario,
            train,
            overwrite=True,
            logging_level=50,
        )

        incumbent = smac.optimize()
        classifier = model.fit(incumbent, train_x, train_y, seed=12345)
        train_time = time.perf_counter() - t0

        # Evaluate the best configuration on the test set
        t0 = time.perf_counter()
        test_x[:] = scaler.transform(test_x)
        pred = classifier.predict(test_x)
        pred = encoder.inverse_transform(pred.astype(int))
        test_time = time.perf_counter() - t0
        proba = classifier.predict_proba(test_x)

        proba = fix_probabilities(np.unique(test_y), np.unique(train_y), proba)
        if proba.shape[1] == 2:
            proba = proba[:, 1]

        scores = score(test_y, pred, proba)
        scores.update(
            {
                **info,
                "model": f"{bo_name}_{model_type}",
                "dataset": dataset_name,
                "train_time": train_time,
                "test_time": test_time,
            }
        )

        results.append(scores)

        model.save(classifier, os.path.join(model_path, run_name))

        bar.update(1)

    results = pd.DataFrame(results)
    results.to_parquet(os.path.join(results_path, f"{bo_name}_{model_type}.parquet"))


if __name__ == "__main__":
    main()
