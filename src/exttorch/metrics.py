# Praise Ye The Lord

# Import libraries
import numpy as np
import torch
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, List, Callable, Dict
from dataclasses import dataclass
from .__data_handle import SinglePredictionsFormat

class Logs:
    def __init__(self) -> None:
        self.step = 0
        self.logs = {}

    def create_step(self):
        # Create the key in the logs
        self.logs[f'step_{self.step}'] = {
            "model": None,
            "feature": [],
            "label": [],
            "loss": []
        }

class LossStorage:
    """
    Class for storing losses
    """
    def __init__(self):
        self.__loss = []

    @property
    def loss(self) -> float:
        """
        Returns the average loss
        """
        return round(np.array(self.__loss).mean(), 4)

    @loss.setter
    def loss(self, loss) -> None:
        """
        Sets the loss
        Parameters
        ----------
        loss : float
            Loss value
        """
        self.__loss.append(loss)


class Metric(metaclass=ABCMeta):
    """
    Abstract class for metrics
    """
    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        ...


class Accuracy(Metric):
    """
    Class for measuring accuracy metric
    """
    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the metric, by default None
        """
        self.name = 'Accuracy' if name is None else name

    def __str__(self) -> str:
        """
        Returns the name of the metric
        """
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values
        y : torch.Tensor
            True values
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, prediction)


class ZeroOneLoss(Metric):
    """
    Class for measuring zero-one loss metric
    """
    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the metric, by default None
        """
        self.name = 'ZeroOneLoss' if name is None else name

    def __str__(self) -> str:
        """
        Returns the name of the metric
        """
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values
        y : torch.Tensor
            True values
        """
        from sklearn.metrics import zero_one_loss
        return zero_one_loss(y, prediction)

class F1Score(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        self.__kwargs = kwargs
        self.name = 'F1Score' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import f1_score
        return f1_score(y, prediction, **self.__kwargs)

class MatthewsCorrcoef(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MatthewsCorrcoef' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y, prediction, **self.__kwargs)

class Recall(Metric):
    def __init__(self, name: Optional[str] = None,  **kwargs):
        self.__kwargs = kwargs
        self.name = 'Recall' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import recall_score
        return recall_score(prediction, y, **self.__kwargs)

class Jaccard(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Jaccard' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import jaccard_score
        return jaccard_score(prediction, y, **self.__kwargs)

class Precision(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Precision' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import precision_score
        return precision_score(prediction, y, **self.__kwargs)

def handle_probability(proba):
    from torch.nn import functional as f

    if proba.shape[1] > 2:
        _proba = (proba.clone().detach()
                    if type(proba) != np.ndarray
                    else torch.tensor(proba)
                    )
        return f.softmax(_proba, dim=-1)
    return proba[:, 0]


class TopKAccuracy(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'TopKAccuracy' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, y: torch.Tensor, proba: torch.Tensor) -> Any:
        from sklearn.metrics import top_k_accuracy_score

        proba = handle_probability(proba)
        return top_k_accuracy_score(y, proba, **self.__kwargs)

class Auc(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Auc' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, y: torch.Tensor, proba: torch.Tensor) -> Any:
        from sklearn.metrics import roc_auc_score

        proba = handle_probability(proba)
        return roc_auc_score(y, proba, **self.__kwargs)


class MeanSquaredError(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MeanSquaredError' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(prediction, y.reshape(y.shape[0], 1))


class MeanAbsoluteError(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MeanAbsoluteError' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(prediction, y.reshape(y.shape[0], 1),
                                   **self.__kwargs)

class R2(Metric):
    def __init__(self, name: Optional[str] = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'R2' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor) -> Any:
        from sklearn.metrics import r2_score
        return r2_score(
            prediction.view(prediction.shape[0], 1),
            y.reshape(y.shape[0], 1),
                        **self.__kwargs)


def str_val_to_metric(metric_list: List[Metric | str]):
    new_metric_list = []
    for new_metric_name in metric_list:
        if type(new_metric_name) == str:
            match new_metric_name:
                case 'acc' | 'Acc' | 'accuracy' | 'Accuracy':
                    new_metric_list.append(Accuracy(new_metric_name))
                case 'mse' | 'MSE' | 'MeanSquaredError':
                    new_metric_list.append(MeanSquaredError(new_metric_name))
                case 'r2' | 'R2':
                    new_metric_list.append(R2(new_metric_name))
                case 'mae' | 'MAE' | 'MeanAbsoluteError':
                    new_metric_list.append(MeanAbsoluteError(new_metric_name))
                case 'recall' | 'rec' | 'Recall':
                    new_metric_list.append(Recall(new_metric_name))
                case 'precision' | 'pre' | 'Precision':
                    new_metric_list.append(Precision(new_metric_name))
                case 'jaccard' | 'jac' | 'Jaccard':
                    new_metric_list.append(Jaccard(new_metric_name))
                case 'Auc' | 'auc':
                    new_metric_list.append(Auc(new_metric_name))
                case 'MatthewsCorrcoef' | 'mat' | 'mc' | 'MC':
                    new_metric_list.append(MatthewsCorrcoef(new_metric_name))
                case 'ZeroOneLoss' | 'zero' | 'zol':
                    new_metric_list.append(ZeroOneLoss(new_metric_name))
                case 'TopKAccuracy' | 'TKA' | 'tka':
                    new_metric_list.append(TopKAccuracy(new_metric_name))
                case _:
                    raise ValueError(f'Unknown metric name `{new_metric_name}`')
        else:
            new_metric_list.append(new_metric_name)

    return new_metric_list



@dataclass
class MetricComputation:
    metric: Callable
    prediction: torch.Tensor | np.intp | np.ndarray
    y: Any

    def compute_metric(self):
        if len(self.y.shape) >= 1:
            return self.metric(
                self.prediction,
                self.y)

        # Change to cpu if it's a Tensor
        predictions = (self.prediction.cpu().numpy()
            if isinstance(self.prediction, torch.Tensor)
            else self.prediction)

        return self.metric(predictions, torch.tensor([self.y]).numpy())

class MetricStorage:
    def __init__(self,
                metrics: List[Metric],
                batch_size: Optional[int] = None):
        self.__labels = []
        self.__metrics = metrics
        self.__metric_dict = {str(metric): [] for metric in metrics}
        self.__batch_size = batch_size
        self.__metric_name_proba = ['Auc', 'TopKAccuracy', 'auc', 'tka', 'TKA']

    def __y_prediction_or_proba(self, metric: Metric, predict, formatted_prediction):

        # Return formatted prediction or probability
        return (
            formatted_prediction
            if str(metric) not in self.__metric_name_proba
            else predict.detach().cpu().numpy()
        )


    def add_metric(self,
                predict,
                label,
                ) -> None:

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(predict)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        if self.__batch_size is not None and self.__batch_size > 1:
            # Change batched prediction to one single metric e.g 0.983.
            # Loop over the metrics
            for metric in self.__metrics:
                self.__metric_dict[str(metric)].append(
                    MetricComputation(
                        metric,
                        label,
                        self.__y_prediction_or_proba(metric, predict, formatted_prediction))
                    .compute_metric())
        else:
            # Add single prediction
            # Loop over the metrics and metric names
            for metric in self.__metrics:
                self.__metric_dict[str(metric)].append(
                    self.__y_prediction_or_proba(metric, predict[0], formatted_prediction)
                    )
            # Save labels in the list
            self.__labels.append(label)

    def metrics(self, y: Any = None):
        if self.__batch_size is not None and self.__batch_size > 1:
            # Return a new dictionary with list mean
            return {
                key: round(torch.tensor(value).mean().item(), 4)
                for key, value in self.__metric_dict.items()
            }
        _metric = {str(metric): metric  for metric in self.__metrics}
        metrics_dict = {}

        for key, value in self.__metric_dict.items():
            values = torch.tensor(
                self.__handle_values_from_metric_dict(value),
                dtype=torch.float64
                )
            y = y if y is not None else torch.tensor(self.__labels)
            metric_comp = MetricComputation(
                            _metric[key],
                            y, values
                        ).compute_metric()

            metrics_dict[key] = round(
                (metric_comp.item()
                if type(metric_comp) != float
                else metric_comp), 4)

        return metrics_dict
    @staticmethod
    def __handle_values_from_metric_dict(value):
        # try:
        return np.array([
                val.clone().detach().cpu().numpy()
                if type(val) == torch.Tensor
                else val
                for val in value], dtype=np.float64)

def change_metric_first_position(measurements) -> Dict:
    keys = list(measurements.keys())
    loss_idx = keys.index('loss')
    keys.pop(loss_idx)
    keys.insert(0, 'loss')
    measurements = {
        key: measurements[key]
        for key in keys
    }
    return measurements
