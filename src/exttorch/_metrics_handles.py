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
    F1Score
)


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


class SinglePredictionsFormat:
    def __init__(self, prediction, device, loss_name):
        import numpy as np
        self.__np = np
        self.__prediction = prediction
        self.__device = device
        self.___main_loss_name = loss_name
        self.__size = (
            prediction.size()
            if isinstance(prediction, torch.Tensor)
            else prediction.shape
        )

    def __single_format(self, prediction):
        if self.___main_loss_name in ["BCELoss", "BCEWithLogitsLoss"]:
            # It's a binary classification
            return self.__np.round(prediction).astype(int)
        elif self.___main_loss_name in ["CrossEntropyLoss", "NLLLoss"]:
            # It's a multi-class classification
            return self.__np.argmax(prediction)
        
        # else it's a continuous prediction
        return prediction

    def format_prediction(self) -> Any:
        if self.__size[0] > 1:
            # It's a batched prediction
            return self.__batched_prediction()
        # It's a single prediction
        return self.__single_format(self.__prediction)

    def __batched_prediction(self) -> torch.Tensor:
        return self.__np.array(
            list(map(lambda tensor: self.__single_format(tensor), self.__prediction)),
        ).reshape(-1, 1)



class MetricStorage:
    def __init__(self, device: str, metrics: list, batch_size: int, loss_name: str=None,  train: bool = True):
        import numpy as np
        
        self.__device = device
        self.__np = np
        self.__metrics = metrics
        self.__metric_dict: Dict[str, np.ndarray[float]] = {}
        self.__train = train
        self.__loss_name = "loss" if train else "val_loss"
        self.__metric_dict[self.__loss_name] = np.array([])
        self.__metric_dict.update({
            (str(metric) if train else "val_"+str(metric)): np.array([]) 
            for metric in metrics
        })
        self.__predicts: np.ndarray = np.array([])
        self.__probabilities: np.ndarray = np.array([])
        self.__labels: np.ndarray = np.array([])
        self.__loss: np.ndarray = np.array([])
        self.__batch_size = batch_size
        self.__main_loss_name = loss_name
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
        single_format_prediction = SinglePredictionsFormat(predict, self.__device, self.__main_loss_name)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()
        self.__predicts = self.__np.append(self.__predicts, formatted_prediction)
        self.__probabilities = self.__np.append(self.__probabilities, handle_probability(predict))
        self.__labels = self.__np.append(self.__labels, label)
        self.__loss = self.__np.append(self.__loss, loss)
                
    def measurements_compiler(self) -> List[Tuple[str, float]]:
        _metric = {(str(metric) if self.__train else "val_"+str(metric)): metric for metric in self.__metrics}
        over_all_metrics: List[Tuple[str, float]] = []
                
        if self.__batch_size > 1 and len(self.__predicts.shape) != 1:
            predications = list(map(lambda x: x.reshape(-1, 1), self.__predicts))
            probability = list(map(lambda x: x.reshape(-1, 1), self.__probabilities))
            labels = list(map(lambda x: x.reshape(-1, 1), self.__labels))
            loss = round(self.__loss.mean(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name] = np.append(self.__metric_dict[self.__loss_name], loss)
            
            _inner_metrics = {
                (str(metric) if self.__train else "val_"+str(metric)): np.array([]) 
                for metric in self.__metrics
                }
            
            for predict, label, prob in zip(predications, labels, probability):
                for key, value in _metric.items():
                    if key.replace("val_", "") in self.__metric_name_proba:
                        metric_comp = MetricComputation(
                            value, predictions=prob.reshape(-1, 1), labels=label.reshape(-1, 1)
                        ).compute_metric(self.__batch_size)
                        metric_comp = 0.0 if math.isnan(metric_comp) or math.isinf(metric_comp) else metric_comp
                    else:
                        metric_comp = MetricComputation(
                            value, predictions=predict, labels=label
                        ).compute_metric(self.__batch_size)
                    
                    _inner_metrics[key] = np.append(_inner_metrics[key], metric_comp)
            
            _inner_metrics = {
                key: value.mean().round(4)
                for key, value in _inner_metrics.items()
            }
            lst = list(_inner_metrics.items())
            over_all_metrics.extend(lst)
            
            for key, value in _inner_metrics.items():
                self.__metric_dict[key] = self.__np.append(self.__metric_dict[key], value)
                
            return over_all_metrics
        else:
            predications = self.__predicts.reshape(-1, 1)
            probability = self.__probabilities
            labels = self.__labels
            loss = round(self.__loss.mean(), 4)
            
            over_all_metrics.append((self.__loss_name, loss))
            self.__metric_dict[self.__loss_name] = np.append(self.__metric_dict[self.__loss_name], loss)
            
            for key, value in _metric.items():
                if key.replace("val_", "") in self.__metric_name_proba:
                    metric_comp = MetricComputation(
                        value, predictions=probability.reshape(-1, 1), labels=labels.reshape(-1, 1)
                    ).compute_metric(self.__batch_size)
                    metric_comp = 0.0 if math.isnan(metric_comp) or math.isinf(metric_comp) else metric_comp
                else:
                    metric_comp = MetricComputation(
                        value, predictions=predications, labels=labels
                    ).compute_metric(self.__batch_size)
                            
                over_all_metrics.append((key, metric_comp))
                self.__metric_dict[key] = self.__np.append(self.__metric_dict[key], metric_comp)
                
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
            k: round(v.mean(), 4)
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
                case "F1Score" | "f1" | "f1score" | "F1":
                    new_metric_list.append(F1Score(new_metric_name))
                case _:
                    raise ValueError(f"Unknown metric name `{new_metric_name}`")
        else:
            new_metric_list.append(new_metric_name)

    return new_metric_list


def handle_probability(proba: torch.Tensor):
    if proba.shape[1] > 2:
        return f.softmax(torch.tensor(proba), dim=-1)[:, 1]
    return proba
