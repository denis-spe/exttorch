# Praise Ye The Lord

# Import libraries
import torch, math
from torch.nn import functional as f
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple
import numpy as np
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


class SinglePredictionsFormat:
    def __init__(self, prediction, device):
        self.__prediction = prediction
        self.__device = device
        self.__size = (
            prediction.size()
            if isinstance(prediction, torch.Tensor)
            else prediction.shape
        )

    def __single_format(self, prediction):
        if self.__size[1] > 1:
            # That's a category prediction
            return torch.argmax(prediction)
                
        # else it's a continuous prediction
        return prediction

    def format_prediction(self) -> Any:
        if self.__size[0] > 1:
            # It's a batched prediction
            return self.__batched_prediction()
        # It's a single prediction
        return self.__single_format(self.__prediction)

    def __batched_prediction(self) -> torch.Tensor:
        return torch.tensor(
            list(map(lambda tensor: self.__single_format(tensor), self.__prediction)),
            device=self.__device
        ).view(-1, 1)



class MetricStorage:
    def __init__(self, device: str, metrics: list, batch_size: int, train: bool = True):
        
        self.__device = device
        self.__metrics = metrics
        self.__metric_dict: Dict[str, List[float]] = {}
        self.__train = train
        self.__loss_name = "loss" if train else "val_loss"
        self.__metric_dict[self.__loss_name] = []
        self.__metric_dict.update({
            (str(metric) if train else "val_"+str(metric)): [] 
            for metric in metrics
        })
        self.__predicts: List[torch.Tensor] = []
        self.__probabilities: List[torch.Tensor] = []
        self.__labels: List[torch.Tensor] = []
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
        single_format_prediction = SinglePredictionsFormat(predict, self.__device)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        self.__predicts.append(formatted_prediction)
        self.__probabilities.append(handle_probability(predict))
        self.__labels.append(label)
        self.__loss.append(loss)
        
    def measurements_compiler(self) -> List[Tuple[str, float]]:
        _metric = {(str(metric) if self.__train else "val_"+str(metric)): metric for metric in self.__metrics}
        over_all_metrics: List[Tuple[str, float]] = []
        
        if self.__batch_size > 1 or len(self.__predicts[0].shape) == 2:
            predications = list(map(lambda x: x.to(self.__device).reshape(-1, 1), self.__predicts))
            probability = list(map(lambda x: x.to(self.__device).reshape(-1, 1), self.__probabilities))
            labels = list(map(lambda x: x.to(self.__device).reshape(-1, 1), self.__labels))
            loss = round(torch.tensor(self.__loss, device=self.__device).mean().item(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name].append(loss)
            
            _inner_metrics = {
                (str(metric) if self.__train else "val_"+str(metric)): [] 
                for metric in self.__metrics
                }
            
            for predict, label, prob in zip(predications, labels, probability):
                for key, value in _metric.items():
                    if key.replace("val_", "") in self.__metric_name_proba:
                        metric_comp = MetricComputation(
                            value, predictions=prob.view(-1), labels=label.view(-1)
                        ).compute_metric(self.__batch_size)
                        metric_comp = 0.0 if math.isnan(metric_comp) or math.isinf(metric_comp) else metric_comp
                    else:
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
            probability = torch.tensor([self.__probabilities], device=self.__device)
            # print(self.__labels)
            labels = torch.tensor(self.__labels, device=self.__device)
            loss = round(torch.tensor(self.__loss, device=self.__device).mean().item(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name].append(loss)
            
            for key, value in _metric.items():
                if key.replace("val_", "") in self.__metric_name_proba:
                    metric_comp = MetricComputation(
                        value, predictions=probability.view(-1), labels=labels.view(-1)
                    ).compute_metric(self.__batch_size)
                    metric_comp = 0.0 if math.isnan(metric_comp) or math.isinf(metric_comp) else metric_comp
                else:
                    metric_comp = MetricComputation(
                        value, predictions=predications, labels=labels
                    ).compute_metric(self.__batch_size)
                            
                over_all_metrics.append((key, metric_comp))
                self.__metric_dict[key].append(metric_comp)
                
            over_all_metrics = list(map(lambda x: (x[0], self.__between_zero_and_one(x[0], x[1])), over_all_metrics))
            return over_all_metrics
    
    def __between_zero_and_one(self, name: str, value: float) -> float:
        match name:
            case "Auc" | "auc" | "val_Auc" | "val_auc":
                return max(0.0, min(value, 1.0))
            case _:
                return value
            

    @property
    def measurements(self) -> Dict[str, float]:
        
        # # Alter metric_dict values (list) to mean
        altered_list_to_mean_dict = {
            k: round(torch.tensor(v).mean().item(), 4) 
            for k, v in self.__metric_dict.items()
        }
        
        altered_list_to_mean_dict = list(map(
            lambda x: (x[0], self.__between_zero_and_one(x[0], x[1])), 
            altered_list_to_mean_dict.items()))
        return dict(altered_list_to_mean_dict)


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
        return f.softmax(proba, dim=-1)[:, 1]
    return proba[:, 1]
