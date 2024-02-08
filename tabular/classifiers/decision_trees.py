import os
from typing import Union

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Integer
from joblib import dump
from sklearn import tree
from sklearn.base import ClassifierMixin

from .utils import ClassifierModule


class DecisionTree(ClassifierModule):
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ) -> ClassifierMixin:
        cs = config.get_dictionary()
        if cs["class_weight"] == "None":
            cs["class_weight"] = None
        classifier = tree.DecisionTreeClassifier(**cs, random_state=seed)
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
    def parameters(**kwargs) -> ConfigurationSpace:
        configspace = ConfigurationSpace(
            name="Decision Tree Classifier",
            space={
                "max_depth": (2, 10),
                "splitter": ["best", "random"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": Integer(
                    "min_samples_split", bounds=(2, 20), log=True
                ),
                "min_samples_leaf": Integer(
                    "min_samples_leaf", bounds=(1, 20), log=True
                ),
                "max_features": Integer("max_features", bounds=(1, 10), log=True),
                "class_weight": ["balanced", "None"],
            },
            meta=dict(
                indexnames=[
                    "Maximum Tree Depth",
                    "Splitting Strategy",
                    "Split Quality Criterion",
                    "Min. Samples to Split Internal Node",
                    "Min. Samples to Split Leaf Node"
                    "Features Considered During Split",
                    "Feasible Sample Weight",
                ]
            ),
        )

        return configspace

    @staticmethod
    def save(model, path):
        model_path = os.path.join(path, "model.joblib")

        # Save model
        dump(model, model_path)
