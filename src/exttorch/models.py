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

    def predict_proba(self, x, verbose=None):
        import torch
        import numpy as np
        from .utils import ProgressBar
        from torch.nn import functional as f

        x = (x.float() if type(x) == torch.Tensor else torch.tensor(x).float()).to(
            self._device
        )
        
        if self.model is None:
            raise ValueError("The model is not trained yet. Please train the model first.")

        # Instantiate the progress bar
        progressbar = ProgressBar(
            bar_width=self._progress_bar_width,
            fill_style=self._progress_fill_style,
            empty_style=self._progress_empty_style,
            fill_color=self._progress_fill_color,
            empty_color=self._progress_empty_color,
            percentage_colors=self._progress_percentage_colors,
            progress_type=self._progress_progress_type,
            verbose=self._verbose if verbose is None else verbose,
        )
        # Set the progress bar total
        progressbar.total = len(x)

        # Empty list for probability
        probability = []

        with torch.no_grad():
            for i, data in enumerate(x):

                # Make prediction and get probabilities
                proba = self.model(data)

                # Append the probabilities to the list
                probability.append(proba.detach().reshape(1, -1).tolist()[0])

                # Update the progress bar
                progressbar.update(i + 1)

        if type(self.loss_obj).__name__ == "CrossEntropyLoss":
            probability = f.softmax(torch.tensor(probability), dim=1).numpy()
        else:
            # Convert the probability to numpy array
            probability = np.array(probability).reshape(-1, 1)

        print("\n")

        return probability

    def predict(self, x, verbose=None):

        # Get the probabilities of x
        probability = self.predict_proba(x, verbose=verbose)

        # Get the class label if using CrossEntropyLoss
        # or BCELoss or BCEWithLogitsLoss
        if type(self.loss_obj).__name__ == "CrossEntropyLoss":
            predict = probability.argmax(axis=1).reshape(-1, 1)
        elif type(self.loss_obj).__name__ in ["BCELoss", "BCEWithLogitsLoss"]:
            predict = probability.round().reshape(-1, 1)
        else:
            predict = probability.reshape(-1, 1)
        return predict

    def compile(
        self,
        optimizer: __Optimizer__ | str,
        loss: __Loss__ | str,
        metrics: __tp__.List[str | __Metric__] | None = None,
    ):
        """
        Compile the model.

        Parameters
        ----------
            optimizer : (Optimizer | str)
                For updating the model parameters.
                Here are some of the options:
                    - Adam: Adam optimizer
                    - SGD: Stochastic Gradient Descent
                    - RMSprop: RMSprop optimizer
                    - Adadelta: Adadelta optimizer
            loss : (Loss | str)
                Measures model's performance.
                Here are some of the options:
                    - BCELoss: Binary Cross Entropy Loss
                    - BCEWithLogitsLoss: Binary Cross Entropy Loss with Logits
                    - CrossEntropyLoss: Cross Entropy Loss
                    - MSELoss: Mean Squared Error Loss
                    - NLLLoss: Negative Log Likelihood Loss
            metrics : (Optional[List[Metric|str]]) default
                Measures model's performance.
                Here are some of the options:
                    - Accuracy: A classification metric for measuring model accuracy.
                    - F1Score: A classification metric for measuring model f1 score.
                    - MAE: Mean absolute error for regression problem.
                    - MSE: Mean squared error for regression problem.
                    - AUC: Area under the curve for classification problem.
                    - Recall: A classification metric for measuring model recall score.
                    - Precision: A classification metric for measuring model precision score.

        """
        self.optimizer_obj = (
            optimizer
            if type(optimizer) != str
            else self.__change_str_to_optimizer__(optimizer)
        )
        self.loss_obj = loss if type(loss) != str else self.__change_str_to_loss__(loss)
        self.metrics = (
            self.__str_val_to_metric__(metrics) if metrics is not None else []
        )

    @staticmethod
    def __str_val_to_metric__(
        metric_list: __tp__.List[__tp__.Any],
    ) -> __tp__.List[__Metric__]:
        from src.exttorch.metrics import (
            Accuracy,
            MeanSquaredError,
            R2,
            MeanAbsoluteError,
            Recall,
            Precision,
            Jaccard,
            Auc,
            MatthewsCorrcoef,
            ZeroOneLoss,
            TopKAccuracy,
            F1Score,
        )

        new_metric_list: __tp__.List[__Metric__] = []
        for new_metric_name in metric_list:
            if type(new_metric_name) == str:
                match new_metric_name:

                    case "acc" | "Acc" | "accuracy" | "Accuracy":
                        new_metric_list.append(Accuracy(new_metric_name))
                    case "mse" | "MSE" | "MeanSquaredError":
                        new_metric_list.append(MeanSquaredError(new_metric_name))
                    case "r2" | "R2":
                        new_metric_list.append(R2(new_metric_name))
                    case "mae" | "MAE" | "MeanAbsoluteError":
                        new_metric_list.append(MeanAbsoluteError(new_metric_name))
                    case "recall" | "rec" | "Recall":
                        new_metric_list.append(Recall(new_metric_name))
                    case "precision" | "pre" | "Precision":
                        new_metric_list.append(Precision(new_metric_name))
                    case "jaccard" | "jac" | "Jaccard":
                        new_metric_list.append(Jaccard(new_metric_name))
                    case "Auc" | "auc":
                        new_metric_list.append(Auc(new_metric_name))
                    case "MatthewsCorrcoef" | "mat" | "mc" | "MC":
                        new_metric_list.append(MatthewsCorrcoef(new_metric_name))
                    case "ZeroOneLoss" | "zero" | "zol":
                        new_metric_list.append(ZeroOneLoss(new_metric_name))
                    case "TopKAccuracy" | "TKA" | "tka":
                        new_metric_list.append(TopKAccuracy(new_metric_name))
                    case "F1Score" | "f1" | "f1score" | "F1" | "f1_score":
                        new_metric_list.append(F1Score(new_metric_name))
                    case _:
                        raise ValueError(f"Unknown metric name `{new_metric_name}`")
            else:
                new_metric_list.append(new_metric_name)

        return new_metric_list

    @staticmethod
    def __change_str_to_loss__(loss: str):
        from src.exttorch.losses import (
            MSELoss,
            L1Loss,
            NLLLoss,
            CrossEntropyLoss,
            BCELoss,
            BCEWithLogitsLoss,
            MarginRankingLoss,
        )

        match loss:
            case "MSELoss" | "mse" | "mean_squared_error" | "MSE":
                return MSELoss()
            case "L1Loss" | "l1" | "mean_absolute_error" | "MAE":
                return L1Loss()
            case "NLLLoss" | "nll" | "negative_log_likelihood" | "nll_loss":
                return NLLLoss()
            case (
                "CrossEntropyLoss"
                | "cross_entropy"
                | "crossentropy"
                | "categorical_crossentropy"
            ):
                return CrossEntropyLoss()
            case "BCELoss" | "bce" | "binary_crossentropy":
                return BCELoss()
            case (
                "BCEWithLogitsLoss"
                | "bce_with_logits"
                | "binary_cross_entropy_with_logits"
            ):
                return BCEWithLogitsLoss()
            case "MarginRankingLoss" | "margin_ranking":
                return MarginRankingLoss()
            case _:
                raise ValueError(
                    "Invalid loss name. Available options: "
                    "MSELoss, L1Loss, NLLLoss, CrossEntropyLoss, "
                )

    @staticmethod
    def __change_str_to_optimizer__(optimizer: str):
        from src.exttorch.optimizers import (
            Adam,
            SGD,
            RMSprop,
            Adadelta,
            Adagrad,
            Adamax,
            ASGD,
        )

        match optimizer:
            case "Adam" | "adam":
                return Adam()
            case "SGD" | "sgd":
                return SGD()
            case "RMSprop" | "rmsprop":
                return RMSprop()
            case "Adadelta" | "adadelta":
                return Adadelta()
            case "Adagrad" | "adagrad":
                return Adagrad()
            case "Adamax" | "adamax":
                return Adamax()
            case "ASGD" | "asgd":
                return ASGD()
            case _:
                raise ValueError(
                    f"Invalid optimizer name `{optimizer}`. Available options: "
                    "Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, ASGD."
                )


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
