# Praise Ye The Lord

# Import libraries
import torch
from torch.nn import functional as f
from dataclasses import dataclass
from typing import Callable, Any, Dict
import numpy as np
from exttorch._data_handle import SinglePredictionsFormat
from exttorch.metrics import (
    Accuracy,
    MeanSquaredError,
    R2,
    MeanAbsoluteError,
    Recall,
    Precision,
    Jaccard,
    MatthewsCorrcoef,
    Auc,
    ZeroOneLoss,
    TopKAccuracy,
)


class Logs:
    def __init__(self) -> None:
        self.step = 0
        self.logs = {}

    def create_step(self):
        # Create the key in the logs
        self.logs[f"step_{self.step}"] = {
            "model": None,
            "feature": [],
            "label": [],
            "loss": [],
        }


@dataclass
class MetricComputation:
    metric: Callable
    prediction: torch.Tensor | np.intp | np.ndarray
    y: Any

    def compute_metric(self):

        if len(self.y.shape) >= 1:
            return self.metric(
                self.prediction,
                self.y
            )

        return self.metric(self.prediction, torch.tensor([self.y]))


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


class MetricStorage:
    def __init__(self, metrics: list, batch_size: int = None):
        self.__labels = []
        self.__metrics = metrics
        self.__metric_dict = {str(metric): [] for metric in metrics}
        self.__predicts = []
        self.__batch_size = batch_size
        self.__metric_name_proba = ["Auc", "TopKAccuracy", "auc", "tka", "TKA"]

    def __y_prediction_or_proba(self, metric, predict, formatted_prediction):

        # Return formatted prediction or probability
        return (
            formatted_prediction
            if str(metric) not in self.__metric_name_proba
            else predict
        )

    def add_metric(
        self,
        predict,
        label,
    ) -> None:

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(predict)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        self.__predicts.append(formatted_prediction)
        self.__labels.append(label)

    def metrics(self, y: Any = None):
        metrics_dict = {}
        _metric = {str(metric): metric for metric in self.__metrics}

        if y is None:
            y = self.__labels

        # Predictions
        predicts = self.__predicts

        if type(y) == np.ndarray:
            y = y.reshape(-1, 1)
        elif type(y) == torch.Tensor:
            y = y.reshape(-1, 1)

        # Check if the label greater than one.
        if len(y) > 1:
            # Creating a dictionary to store result metric_comp product by batching of data
            metric_dict = {key: [] for key in _metric.keys()}

            # Grouping predict with y in order loop them
            for predict, label in zip(predicts, y):

                # Loop over the metric name as key and metric class as value.
                for key, value in _metric.items():
                    metric_comp = MetricComputation(
                        value, label, predict
                    ).compute_metric()
                    metric_dict[key].append(metric_comp)

            # Alter metric_dict values (list) to mean
            altered_list_to_mean_dict = {
                k: np.mean(v).round(4) for k, v in metric_dict.items()
            }

            # Add (update) new altered_list_to_mean_dict dictionary to metrics_dict dictionary
            metrics_dict.update(altered_list_to_mean_dict)
        return metrics_dict


def change_metric_first_position(measurements) -> Dict:
    keys = list(measurements.keys())
    loss_idx = keys.index("loss")
    keys.pop(loss_idx)
    keys.insert(0, "loss")
    measurements = {key: measurements[key] for key in keys}
    return measurements


def str_val_to_metric(metric_list: list):
    new_metric_list = []
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
                case _:
                    raise ValueError(f"Unknown metric name `{new_metric_name}`")
        else:
            new_metric_list.append(new_metric_name)

    return new_metric_list


def handle_probability(proba):
    if proba.shape[1] > 2:
        _proba = (
            proba.clone().detach() if type(proba) != np.ndarray else torch.tensor(proba)
        )
        return f.softmax(_proba, dim=-1)
    return proba[:, 0]
