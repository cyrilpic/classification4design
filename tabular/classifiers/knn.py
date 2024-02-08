import os
from typing import Union

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Integer
from joblib import dump
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from .utils import ClassifierModule


class KNN(ClassifierModule):
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ) -> ClassifierMixin:
        cs = config.get_dictionary()
        classifier = KNeighborsClassifier(**cs)
        classifier.fit(x_train, y_train)
        return classifier

    @classmethod
    def fit_score(
        cls,
        config: Configuration,
        X: np.ndarray,
        y: np.ndarray,
        train: np.ndarray,
        test: np.ndarray,
        seed: Union[int, np.random.RandomState, None],
    ) -> float:
        classifier = cls.fit(config, X.iloc[train], y.iloc[train], seed=seed)
        return classifier.score(X.iloc[test], y.iloc[test])

    @staticmethod
    def parameters(**kwargs):
        n_samples = int(kwargs.pop("n_samples", 2000) * 0.8)

        configspace = ConfigurationSpace(
            name="KNN Classifier",
            space={
                "n_neighbors": Integer(
                    "n_neighbors", bounds=(1, min(n_samples, 1000)), log=True
                ),
                "weights": Categorical("weights", ["uniform", "distance"]),
                "metric": Categorical(
                    "metric", ["euclidean", "manhattan", "chebyshev"]
                ),
            },
            meta=dict(
                indexnames=[
                    "Number of Neighbors",
                    "Neighbor Weighting",
                    "Distance Metric",
                ]
            ),
        )
        return configspace

    @staticmethod
    def save(model, path):
        model_path = os.path.join(path, "model.joblib")
        # Save model
        dump(model, model_path)
