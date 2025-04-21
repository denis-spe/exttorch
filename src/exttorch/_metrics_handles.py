# # Praise Ye The Lord

# Import libraries
import torch, math
from torch.nn import functional as f
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple
import numpy as np
from exttorch.metrics import (
    Metric,
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
    F1Score,
)


@dataclass
class MetricComputation:
    metric: Metric
    predictions: torch.Tensor
    labels: torch.Tensor

    def compute_metric(self):
        return self.metric(
            self.predictions,
            self.labels,
        )

class MetricStorage:
    def __init__(
        self,
        device: str,
        metrics_measures: list[Metric],
        batch_size: int,
        loss_name: str = None,
        train: bool = True,
    ):
        self.__device = device
        self.__metrics_measures = metrics_measures
        self.__measurements_dict: Dict[str, torch.Tensor[float]] = {}
        self.__train = train
        self.__loss_name = "loss" if train else "val_loss"
        self.__measurements_dict[self.__loss_name] = []
        self.__measurements_dict.update(
            {
                (str(metric) if train else "val_" + str(metric)): []
                for metric in metrics_measures
            }
        )
        self.__batch_probabilities: torch.Tensor = None
        self.__batch_labels: torch.Tensor = None
        self.__loss: List[torch.Tensor] = []
        self.__label: List[torch.Tensor] = []
        self.__probabilities: List[torch.Tensor] = []
        self.__batch_size = batch_size
        self.__main_loss_name = loss_name
        self.__metric_name_proba = ["Auc", "TopKAccuracy", "auc", "tka", "TKA"]

    # 1) First step is to add the model results
    def add_model_results(
        self,
        probability: torch.Tensor,
        label: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Adds the model results to the storage.
        Args:
            probability (torch.Tensor): The model predictions.
            label (torch.Tensor): The true labels.
            loss (torch.Tensor): The loss value.
        """
        # Change the shape of the inputs
        probability, label, loss = self.__change_shape(probability, label, loss)
        
        # Validate the inputs
        self.__validate_dtype(probability, label, loss)
        self.__validator_shape(probability, label, loss)
        
        clone = lambda x: x.clone().detach().cpu().numpy()
        # Convert the inputs to numpy arrays
        probability = clone(probability)
        label = clone(label)
        loss = clone(loss).item()
        
        # Store the results
        if self.__batch_size > 1:
            self.__batch_probabilities = np.atleast_2d(probability)
            self.__batch_labels = np.atleast_2d(label)
        else:
            self.__label.append(label)
            self.__probabilities.append(probability)
        self.__loss.append(loss)

    def __change_shape(
        self,
        probability: torch.Tensor,
        label: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Changes the shape of the model results.
        Args:
            probability (torch.Tensor): The model predictions.
            label (torch.Tensor): The true labels.
            loss (torch.Tensor): The loss value.
        """
        if self.__batch_size == 1:
            if probability.shape[1] > 1:
                probability = probability.squeeze()
            else:
                probability = probability.view(-1, 1)
        return probability, label.view(-1, 1), loss.view(1, -1)

    def __validate_dtype(
        self,
        probability: torch.Tensor,
        label: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Validates the model results.
        Args:
            probability (torch.Tensor): The model predictions.
            label (torch.Tensor): The true labels.
            loss (torch.Tensor): The loss value.
        Raises:
            TypeError: If the inputs are not of type torch.Tensor.
        """
        if not isinstance(probability, torch.Tensor):
            raise TypeError("The probability must be a torch.Tensor.")
        if not isinstance(label, torch.Tensor):
            raise TypeError("The label must be a torch.Tensor.")
        if not isinstance(loss, torch.Tensor):
            raise TypeError("The loss must be a torch.Tensor.")

    def __validator_shape(
        self,
        probability: torch.Tensor,
        label: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Validates the shape of the model results.
        Args:
            probability (torch.Tensor): The model predictions.
            label (torch.Tensor): The true labels.
            loss (torch.Tensor): The loss value.
        Raises:
            ValueError: If the shapes of the inputs are not compatible.
        """
        # if probability.shape[0] != label.shape[0]:
        #     raise ValueError("The probability and label must have the same rows.")
        if loss.shape[0] != 1 and loss.shape[1] != 1:
            raise ValueError("The loss must be a scalar.")

    def __change_list_to_numpy(
        self, value: List[torch.Tensor] | float
    ) -> torch.Tensor:
        """
        Changes the list of tensors to a tensor.
        Args:
            tensor_list (List[torch.Tensor]): The list of tensors.
        Returns:
            torch.Tensor: The tensor.
        """
        return np.array(value).astype(np.float32)
        # if isinstance(value[0], float):
        #     return torch.tensor(value, device=self.__device)
        # return torch.cat(value, dim=0).to(self.__device).view(-1, 1)

    def __change_results_to_tensor(self) -> Tuple[torch.Tensor]:
        """
        Changes the model results to a tensor.
        Returns:
            Tuple[torch.Tensor]: The model results as a tensor.
        """
        
        loss = self.__change_list_to_numpy(self.__loss)

        if self.__batch_size > 1:
            labels = self.__batch_labels
            probabilities = self.__batch_probabilities
        else:
            labels = self.__change_list_to_numpy(self.__label)
            probabilities = self.__change_list_to_numpy(self.__probabilities)
            
        return probabilities, labels, loss

    def __get_predicts(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Gets the predictions of the model.
        Args:
            probabilities (torch.Tensor): The model predictions.
        Returns:
            torch.Tensor: The predictions of the model.
        """
        if self.__main_loss_name in ["BCELoss", "BCEWithLogitsLoss"]:
            # It's a binary classification
            return np.round(probabilities).astype(int)
        elif self.__main_loss_name in ["CrossEntropyLoss", "NLLLoss"]:
            # It's a multi-class classification
            dim = 1 if self.__batch_size > 1 else 0
            # print(probabilities)
            return np.argmax(probabilities, axis=1).reshape(-1, 1)
        # else it's a continuous prediction
        return probabilities

    @staticmethod
    def __average(value: np.ndarray, rounded_by=4) -> float:
        """
        Averages the model results.
        Args:
            value (torch.Tensor): The model results.
            rounded_by (int): The number of decimal points to round by.
        Returns:
            float: The average of the model results.
        """
        return np.mean(value).round(decimals=rounded_by)

    @staticmethod
    def __between_zero_and_one(name: str, value: float) -> float:
        match name:
            case "Auc" | "auc" | "val_Auc" | "val_auc":
                if value == None:
                    return np.array([0.0], dtype=np.float32)
                return np.array([max(0.0, min(value, 1.0))])
            case _:
                return value

    @staticmethod
    def __change_metric_first_position(measurements) -> Dict[str, float]:

        keys = list(measurements.keys())
        name = "loss" if "loss" in keys else "val_loss"
        loss_idx = keys.index(name)
        keys.pop(loss_idx)
        keys.insert(0, name)
        measurements = {key: round(measurements[key], 4) for key in keys}
        return measurements

    def __average_dict_measurements(self):
        """
        Averages the model results.
        Returns:
            Dict[str, float]: The average of the model results.
        """
        # Change the measurements dictionary values to a tensor
        measurements_dict = {
            key: np.concatenate(value)
            for key, value in self.__measurements_dict.items()
        }

        return {key: self.__average(value) for key, value in measurements_dict.items()}

    def measurement_computation(self) -> None:
        """
        Computes the model results.
        """
        # Change the model results to a tensor
        probabilities, labels, loss = self.__change_results_to_tensor()
        
        # Get the predictions of the model
        predicts = self.__get_predicts(probabilities)
        
        # Get the loss name
        loss_name = "loss" if self.__train else "val_loss"

        # Add the loss to the measurements dictionary
        self.__measurements_dict[loss_name].append(loss)
        
        if self.__batch_size > 1:
            probabilities = np.atleast_2d(probabilities)
            labels = np.atleast_2d(labels)
        else:
            probabilities = probabilities.squeeze(axis=2)
            labels = labels.squeeze(axis=2)
        

        # Loop over the measurements of metrics
        for measure in self.__metrics_measures:
            # Check if the measure is a string in the list of probability metrics
            if str(measure) in self.__metric_name_proba:
                
                if probabilities.shape[1] > 1:
                    tensor_probabilities = torch.from_numpy(probabilities)
                    probabilities = f.softmax(tensor_probabilities, dim=1).numpy()                    
                    
                # Compute the metric
                metric = MetricComputation(
                    measure,
                    predictions=probabilities,
                    labels=labels,
                ).compute_metric()
                
                # Handle zeros and ones
                metric = self.__between_zero_and_one(str(measure), metric)

            else:
                # Compute the metric
                metric = MetricComputation(
                    measure, predictions=predicts, labels=labels
                ).compute_metric()

            # Handle the metric value
            # metric = 0.0 if math.isnan(metric) or math.isinf(metric) else metric

            # Check if train or validation
            measure_name = str(measure) if self.__train else "val_" + str(measure)

            # Store the metric value
            self.__measurements_dict[measure_name].append(metric)

    @property
    def measurements(self) -> Dict[str, float]:
        """
        Returns the model results.
        Returns:
            List[Tuple[str, float]]: The model results.
        """
        # Average the measurements
        measurements = self.__average_dict_measurements()

        # Change the order of the measurements
        measurements = self.__change_metric_first_position(measurements)

        return measurements

    @staticmethod
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
                    case "F1Score" | "f1" | "f1score" | "F1" | "f1_score":
                        new_metric_list.append(F1Score(new_metric_name))
                    case _:
                        raise ValueError(f"Unknown metric name `{new_metric_name}`")
            else:
                new_metric_list.append(new_metric_name)

        return new_metric_list