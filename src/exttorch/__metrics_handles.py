# # Praise Ye The Lord

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
# Import libraries
import torch
from numpy import ndarray, dtype, floating
from torch import Tensor
from torch.nn import functional as f

from src.exttorch.metrics import (
    Metric,
)


@dataclass
class MetricComputation:
    metric: Metric
    predictions: np.ndarray
    labels: np.ndarray
    
    def reshape(self):
        if self.predictions.ndim == 3:
            self.predictions = self.predictions.reshape(-1, 1)
        if self.labels.ndim == 3:
            self.labels = self.labels.reshape(-1, 1)

    def compute_metric(self):
        # Reshape if necessary
        self.reshape()
        
        return self.metric(
            self.predictions,
            self.labels,
        )


def validate_dtype(
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


def validator_shape(
        loss: torch.Tensor,
) -> None:
    """
    Validates the shape of the model results.
    Args:
        loss (torch.Tensor): The loss value.
    Raises:
        ValueError: If the shapes of the inputs are not compatible.
    """
    if loss.shape[0] != 1 and loss.shape[1] != 1:
        raise ValueError("The loss must be a scalar.")


def change_list_to_numpy(value: List[torch.Tensor] | float) -> ndarray[Any, dtype[Any]]:
    return np.array(value).astype(np.float32)

class MetricStorage:
    def __init__(
        self,
        device: str | torch.device,
        metrics_measures: list[Metric],
        batch_size: int | None,
        loss_name: str | None = None,
        train: bool = True,
    ):
        self.__device = device
        self.__metrics_measures = metrics_measures
        self.__measurements_dict: Dict[str, Any] = {}
        self.__train = train
        self.__loss_name = "loss" if train else "val_loss"
        self.__measurements_dict[self.__loss_name] = []
        self.__measurements_dict.update(
            {
                (str(metric) if train else "val_" + str(metric)): []
                for metric in metrics_measures
            }
        )
        self.__batch_probabilities: Optional[torch.Tensor] = None
        self.__batch_labels: Optional[torch.Tensor] = None
        self.__loss: List[torch.Tensor] = []
        self.__label: List[torch.Tensor] = []
        self.__probabilities: List[torch.Tensor] = []
        self.__batch_size = batch_size
        self.__main_loss_name = loss_name
        self.__is_batched = False
        self.__metric_name_proba = ["Auc", "TopKAccuracy", "auc", "tka", "TKA"]

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
        validate_dtype(probability, label, loss)
        validator_shape(loss)

        clone = lambda x: x.clone().detach().cpu().numpy()
        # Convert the inputs to numpy arrays
        probability = clone(probability)
        label = clone(label)
        loss = clone(loss).item()

        # Store the results
        if (self.__batch_size is not None and self.__batch_size > 1) or label.shape[0] > 1:
            self.__batch_probabilities = np.atleast_2d(probability)
            self.__batch_labels = np.atleast_2d(label)
            self.__is_batched = True
        else:
            self.__label.append(label)
            self.__probabilities.append(probability)
        self.__loss.append(loss)

    def __change_shape(
        self,
        probability: torch.Tensor,
        label: torch.Tensor,
        loss: torch.Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def __change_results_to_tensor(self):
        """
        Changes the model results to a tensor.
        Returns:
            Tuple[np.ndarray]: The model results as a tensor.
        """

        loss = change_list_to_numpy(self.__loss)

        if self.__batch_size > 1 or self.__is_batched:
            labels = self.__batch_labels
            probabilities = self.__batch_probabilities
        else:
            labels = change_list_to_numpy(self.__label)
            probabilities = change_list_to_numpy(self.__probabilities)

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
            return np.argmax(probabilities, axis=1).reshape(-1, 1)
        # else it's a continuous prediction
        return probabilities

    @staticmethod
    def __average(value: np.ndarray, rounded_by=4) -> floating[Any]:
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
    def __between_zero_and_one(name: str, value: float | ndarray) -> float | ndarray[Any, dtype[Any]]:
        match name:
            case "Auc" | "auc" | "val_Auc" | "val_auc":
                if value is None:
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

        # Check if the batch size is greater than 1 and the labels are one-hot encoded
        if self.__batch_size > 1 or self.__is_batched:
            probabilities = np.atleast_2d(probabilities)
            labels = np.atleast_2d(labels)
        else:
            if probabilities.ndim != 2:
                probabilities = probabilities.squeeze(axis=2)

            if labels.ndim != 2:
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

    def update_state(
        self,
        predict,
        label,
        loss,
    ):
        """
        Handle the metrics for the model.
        Parameters
        ----------
            predict : (torch.Tensor)
                The prediction of the model.
            label : (torch.Tensor)
                The label of the model.
            loss : (torch.Tensor)
                The loss of the model.
        """
        # Add the prediction, labels(target) and loss to metric storage
        self.add_model_results(
            predict.detach(),
            label=label.detach(),
            loss=loss.detach(),
        )

        # Measurement live update
        self.measurement_computation()
