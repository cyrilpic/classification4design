import json
from typing import Union

import numpy as np
import pandas as pd
import torch.nn as nn
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    FillMissing,
    IndexSplitter,
    Normalize,
    TabularPandas,
    accuracy,
    range_of,
    tabular_config,
    tabular_learner,
)
from sklearn.base import ClassifierMixin

from .utils import ClassifierModule

activation_dict = {
    "ReLU": nn.ReLU(inplace=True),
    "Leaky ReLU": nn.LeakyReLU(inplace=True),
    "GELU": nn.GELU(),
}


class FastAIClassifier(ClassifierMixin):
    def __init__(self, learn):
        self.learn = learn

    def predict(self, X):
        test_df = pd.DataFrame(X.copy())
        dl = self.learn.dls.test_dl(test_df)
        _, _, dec_preds = self.learn.get_preds(dl=dl, with_decoded=True)
        return dec_preds.numpy()

    def predict_proba(self, X):
        test_df = pd.DataFrame(X.copy())
        dl = self.learn.dls.test_dl(test_df)
        return self.learn.get_preds(dl=dl)[0].numpy()


class FastAI(ClassifierModule):
    @staticmethod
    def fit(
        config: Configuration,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: Union[int, np.random.RandomState, None] = None,
        **kwargs,
    ) -> ClassifierMixin:
        cs = config.get_dictionary()

        x_columns = x_train.columns
        y_column = "target"

        df = x_train.copy()
        df[y_column] = y_train

        if "x_val" in kwargs:
            x_val = kwargs["x_val"]
            y_val = kwargs["y_val"]

            # Create a fastai TabularPandas object
            df_test = x_val.copy()
            df_test[y_column] = y_val

            df = pd.concat([df, df_test], ignore_index=True)
            splits = IndexSplitter(list(range(len(x_train), len(df))))(range_of(df))
        else:
            splits = None

        dl = TabularPandas(
            df,
            procs=[Categorify, FillMissing, Normalize],
            cont_names=x_columns.tolist(),
            y_names=y_column,
            y_block=(CategoryBlock),
            splits=splits,
        )

        config = tabular_config(
            embed_p=cs["emb_drop"],
            ps=cs["ps"],
            act_cls=activation_dict[cs["activation"]],
        )
        dls = dl.dataloaders(bs=cs["bs"], device="cuda")

        metric = accuracy

        learn = tabular_learner(
            dls,
            metrics=metric,
            layers=json.loads(cs["layers"]),
            config=config,
            default_cbs=False,
        )
        learn.model = learn.model.to("cuda")
        learn.fit_one_cycle(cs["epochs"], cs["lr"])
        return FastAIClassifier(learn)

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
            config,
            X.iloc[train],
            y.iloc[train],
            x_val=X.iloc[test],
            y_val=y.iloc[test],
            seed=seed,
        )
        return classifier.score(X.iloc[test], y.iloc[test])

    @staticmethod
    def parameters(**kwargs):
        n_samples = kwargs["n_samples"]

        bs = (16, 32, 64, 128, 256)
        bs = [b for b in bs if b <= n_samples * 0.6]

        spaces = {
            "layers": Categorical(
                "layers",
                (
                    "null",
                    "[200, 100]",
                    "[200]",
                    "[500]",
                    "[1000]",
                    "[500, 200]",
                    "[50, 25]",
                    "[1000, 500]",
                    "[200, 100, 50]",
                    "[500, 200, 100]",
                    "[1000, 500, 200]",
                ),
            ),
            "emb_drop": Float("emb_drop", bounds=(0.0, 0.5), default=0.1),
            "ps": Float("ps", bounds=(0.0, 0.5), default=0.1),
            "bs": Categorical("bs", bs),
            "lr": Float("lr", bounds=(5e-5, 1e-1), default=1e-2, log=True),
            "activation": Categorical("activation", ("ReLU", "Leaky ReLU", "GELU")),
            "epochs": Integer("epochs", bounds=(5, 30), default=30),
        }
        configspace = ConfigurationSpace(
            name="FastAI NN Classifier",
            space=spaces,
            meta=dict(
                indexnames=[
                    "Layers",
                    "Embedding Dropout",
                    "Dropout",
                    "Batch Size",
                    "Learning Rate",
                    "Activation",
                    "Epochs",
                ]
            ),
        )

        return configspace

    @staticmethod
    def save(model, path):
        # Create directory
        model.learn.model_dir = path
        model.learn.save("model")
