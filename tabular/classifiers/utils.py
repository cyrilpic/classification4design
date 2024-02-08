from typing import Union

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.base import ClassifierMixin


class ClassifierModule:
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ) -> ClassifierMixin:
        pass

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
        pass

    @staticmethod
    def parameters(**kwargs) -> ConfigurationSpace:
        pass

    @staticmethod
    def save(model: ClassifierMixin, path: str, **kwargs):
        pass
