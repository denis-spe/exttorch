"""Praise Ye The Lord Your God"""

import torch as __torch__
from src.exttorch import __types as __types__
from src.exttorch.__model import Model, Wrapper # type: ignore


class StackedModel(Model):
    def __init__(self, layers: __types__.Layers=None, device: str = "cpu"):
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
        >>> from src.exttorch.models import StackedModel
        >>> from torch import nn
        >>> from sklearn.datasets import load_iris
        >>>
        >>> # Load the iris dataset
        >>> x, y = load_iris(return_X_y=True)
        >>>
        >>> # Create the model
        >>> model = StackedModel([
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
        return __torch__.load(model_path, weights_only=True)
    else:
        raise ValueError("Filepath must end with .ext or .we")
