# Praise Ye The Lord

# Import libraries
import torch
from torch.nn import functional as f
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple
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
    predictions: torch.Tensor
    labels: torch.Tensor

    def compute_metric(self, data_size: int):    
        return self.metric(
            self.predictions, 
            self.labels,
            size=data_size
            )


class LossStorage:
    """
    Class for storing losses
    """

    def __init__(self, device):
        self.__loss = []
        self.__device = device


    @property
    def loss(self) -> torch.Tensor:
        """
        Returns the average loss
        """
        return torch.round(
            torch.tensor(self.__loss, device=self.__device).mean(), 
            decimals=4
            ).item()

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
    def __init__(self, device: str, metrics: list, batch_size: int, train: bool = True):
        
        self.__device = device
        self.__metrics = metrics
        self.__metric_dict = {}
        self.__loss_name = "loss" if train else "val_loss"
        self.__metric_dict[self.__loss_name] = []
        self.__metric_dict.update({str(metric): [] for metric in metrics})
        self.__predicts = []
        self.__labels = []
        self.__loss = []
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
        loss,
    ) -> None:

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(predict)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        self.__predicts.append(formatted_prediction)
        self.__labels.append(label)
        self.__loss.append(loss)
        
    def measurements_compiler(self) -> List[Tuple[str, float]]:
        _metric = {str(metric): metric for metric in self.__metrics}
        over_all_metrics = []
        
        if self.__batch_size > 1:
            predications = list(map(lambda x: x.to(self.__device).reshape(-1, 1), self.__predicts))
            labels = list(map(lambda x: x.to(self.__device).reshape(-1, 1), self.__labels))
            loss = round(torch.tensor(self.__loss, device=self.__device).mean().item(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name].append(loss)
            
            _inner_metrics = {str(metric): [] for metric in self.__metrics}
            
            for predict, label in zip(predications, labels):
                for key, value in _metric.items():
                    metric_comp = MetricComputation(
                        value, predictions=predict, labels=label
                    ).compute_metric(self.__batch_size)
                    
                    _inner_metrics[key].append(metric_comp)
            
            _inner_metrics = {
                key: round(torch.tensor(value, device=self.__device).mean().item(), 4)
                for key, value in _inner_metrics.items()
            }
            lst = list(_inner_metrics.items())
            over_all_metrics.extend(lst)
            
            for key, value in _inner_metrics.items():
                self.__metric_dict[key].append(value)
                
            return over_all_metrics
        else:
            predications = torch.tensor(self.__predicts, device=self.__device).reshape(-1, 1)
            labels = torch.tensor(self.__labels, device=self.__device).reshape(-1, 1)
            loss = round(torch.tensor(self.__loss, device=self.__device).mean().item(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name].append(loss)
            
            for key, value in _metric.items():
                metric_comp = MetricComputation(
                    value, predictions=predications, labels=labels
                ).compute_metric(self.__batch_size)
                            
                over_all_metrics.append((key, metric_comp))
                self.__metric_dict[key].append(metric_comp)
                
            return over_all_metrics

    @property
    def measurements(self) -> Dict[str, float]:
        
        # # Alter metric_dict values (list) to mean
        altered_list_to_mean_dict = {
            k: round(torch.tensor(v).mean().item(), 4) for k, v in self.__metric_dict.items()
        }
        
        return altered_list_to_mean_dict

    # @staticmethod
    # def __handle_values_from_metric_dict(value):
    #     if type(value[0]) == torch.Tensor and value[0].shape[0] > 1:
    #         return value[0]

    #     return value


def change_metric_first_position(measurements) -> Dict[str, float]:
    keys = list(measurements.keys())
    loss_idx = keys.index("loss")
    keys.pop(loss_idx)
    keys.insert(0, "loss")
    measurements = {key: round(measurements[key], 4) for key in keys}
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


def handle_probability(proba: torch.Tensor):
    if proba.shape[1] > 2:
        return f.softmax(proba, dim=-1)
    return proba[:, 0]
