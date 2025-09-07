"""Praise Ye The Lord Your God"""

import typing as __tp__

# Import libraries
import torch as __torch__
from numpy.typing import ArrayLike as __ArrayLike__
from sklearn.base import (
    BaseEstimator as __BaseEstimator,
    TransformerMixin as __TransformerMixin,
)
from torch import nn as __nn__
from src.exttorch import __types as __types__
from src.exttorch.__data_handle import DataHandler as __DataHandler__
from src.exttorch.__metrics_handles import MetricStorage as __MetricStorage__
from src.exttorch.history import History as __History__
from src.exttorch.losses import Loss as __Loss__
from src.exttorch.metrics import Metric as __Metric__
from src.exttorch.__model import Model
from src.exttorch.optimizers import Optimizer as __Optimizer__
from src.exttorch.utils import ProgressBar as __ProgressBar__


class StackedModel(Model):
    def __init__(self, layers=None, device: str = "cpu"):
        """
        This represents model algorithm for training and predicting data

        Parameters
        -----------
            layers : (list)
                List of torch layers for training the model.

        Example
        --------
        >>> # Import libraries
        >>> import torch
        >>> from exttorch.models import Stack
        >>> from torch import nn
        >>> from sklearn.datasets import load_iris
        >>>
        >>> # Load the iris dataset
        >>> x, y = load_iris(return_X_y=True)
        >>>
        >>> # Create the model
        >>> model = Stack([
        ...    nn.Linear(4, 8),
        ...    nn.ReLU(),
        ...    nn.Linear(8, 3),
        ...    nn.Softmax(dim=1)
        ... ])
        >>>
        >>> # Compile the model
        >>> model.compile(
        ...    optimizer="Adam",
        ...    loss="CrossEntropyLoss",
        ...    metrics=['accuracy']
        ... )
        >>>
        >>> # Fit the model
        >>> history = model.fit(
        ...     x, y,
        ...     epochs=5,
        ...     verbose=None,
        ...     random_seed=42
        ... )
        >>>
        >>> # Evaluate the model
        >>> model.evaluate(x, y, verbose=None) # doctest: +ELLIPSIS
        {'val_loss': ..., 'val_accuracy': ...}
        """
        super().__init__()
        self.layers = [] if layers is None else layers

    def add(self, layer: __types__.Layer):
        self.layers.append(layer)

class Wrapper(__BaseEstimator, __TransformerMixin):
    """
    Wrapper class for exttorch models to make them compatible with sklearn
    """

    def __init__(
        self,
        model: StackedModel,
        loss: __Loss__,
        optimizer: __Optimizer__,
        metrics: __tp__.List[str | __Metric__] | None = None,
        **fit_kwargs,
    ):
        super().__init__()
        self.is_fitted_ = None
        self.model = model
        self.fit_kwargs = fit_kwargs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.history = None

    def fit(self, x, y=None, **kwargs):
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )
        self.history = self.model.fit(
            x, y, **self.fit_kwargs if len(self.fit_kwargs) > 0 else kwargs
        )
        self.is_fitted_ = True
        return self

    def predict(self, x, verbose: str | None = None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "is_fitted_")
        return self.model.predict(x, verbose=verbose)

    def score(self, x, y=None, verbose: str | None = None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "is_fitted_")
        return self.model.evaluate(x, y, verbose=verbose)


def load_model_or_weight(model_path: str):
    """
    Load the model from the given path.

    Parameters
    ----------
        model_path : (str)
            Path to the model file.

    Returns
    -------
        Sequential or Sequential weight
            Loaded model or weights.
    """
    import pickle

    if model_path.endswith(".ext"):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    elif model_path.endswith(".we"):
        # Load the state_dict
        return __torch__.load(model_path)
    else:
        raise ValueError("Filepath must end with .ext or .we")
