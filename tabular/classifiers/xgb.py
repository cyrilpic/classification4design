import os
from typing import Union

import xgboost as xgb
from sklearn.base import ClassifierMixin
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
import numpy as np

from .utils import ClassifierModule


class XGB(ClassifierModule):
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ) -> ClassifierMixin:
        cs = config.get_dictionary()
        classifier = xgb.XGBClassifier(**cs, random_state=seed)
        classifier.fit(x_train, y_train, verbose=False)
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
        num_classes = kwargs.get("num_classes", 2)
        objective = "binary:logistic" if num_classes == 2 else "multi:softmax"
        if num_classes == 2:
            num_classes = 1  # Binary classification

        configspace = ConfigurationSpace(
            name="XGBoost Classifier",
            space={
                "max_depth": Integer("max_depth", bounds=(2, 8), log=False),
                "reg_lambda": Float("reg_lambda", bounds=(0.1, 100), log=True),
                "eta": Float("eta", bounds=(0.0, 1.0), log=False),
                "gamma": Float("gamma", bounds=(0, 1), log=False),
                "min_child_weight": Float(
                    "min_child_weight", bounds=(0.5, 10), log=True
                ),
                "n_estimators": Integer("n_estimators", bounds=(10, 100), log=False),
                "num_class": num_classes,
                "objective": objective,
            },
            meta=dict(
                indexnames=[
                    "Maximum Tree Depth",
                    "L2 Weight Regularization (lambda)",
                    "Learning Rate (eta)",
                    "Minimum Split Loss (gamma)",
                    "Minimum Child Weight",
                    "Number of boosting rounds",
                    "Number of Classes",
                    "Objective",
                ]
            ),
        )
        return configspace

    @staticmethod
    def save(model, path):
        model_path = os.path.join(path, "model.json")

        # Save model
        model.save_model(model_path)
