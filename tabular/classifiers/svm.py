import os
from typing import Union

from joblib import dump
from sklearn import svm
from sklearn.base import ClassifierMixin
import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float

from .utils import ClassifierModule


class SVM(ClassifierModule):
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        proba: bool = True,
        **kwargs,
    ) -> ClassifierMixin:
        cs = config.get_dictionary()
        if cs["class_weight"] == "None":
            cs["class_weight"] = None
        classifier = svm.SVC(
            **cs, cache_size=2000, probability=proba, random_state=seed
        )
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
        classifier = cls.fit(
            config, X.iloc[train], y.iloc[train], seed=seed, proba=False
        )
        return classifier.score(X.iloc[test], y.iloc[test])

    @staticmethod
    def parameters(**kwargs):
        # Create our configuration space
        configspace = ConfigurationSpace(
            name="SVM Classifier",
            space={
                "C": Float("C", bounds=(0.01, 10.0), default=1.0),
                "kernel": Categorical("kernel", ["linear", "rbf", "sigmoid"]),
                "gamma": Categorical("gamma", ["scale", "auto"]),
                "class_weight": Categorical("class_weight", ["balanced", "None"]),
            },
            meta=dict(
                indexnames=[
                    "Regularization",
                    "SVM Kernel",
                    "Kernel Coefficient",
                    "Class Weight",
                ]
            ),
        )

        return configspace

    @staticmethod
    def save(model, path):
        # Create directory
        model_path = os.path.join(path, "model.joblib")

        # Save model
        dump(model, model_path)
