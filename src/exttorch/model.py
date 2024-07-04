# Praise Ye The Lord

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
# Import libraries
import torch
from tensorflow.keras.utils import Progbar  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from exttorch.__data_handle import DataHandler
from exttorch.metrics import LossStorage, MetricStorage
from exttorch.metrics import Metric, change_metric_first_position
from exttorch.metrics import str_val_to_metric

class Sequential(nn.Module):
    def __init__(self,
                layers: List) -> None:
        super(Sequential, self).__init__()
        self.__device = None
        self.loss = None
        self.optimizer = None
        self.layers = layers
        self.metrics = None

        # Import and use the Sequential object
        from torch import nn as _nn
        self.__model = _nn.Sequential(*self.layers).double()

    def forward(self, X) -> torch.Tensor:
        return self.__model(X)

    def add(self, layer: Any):
        self.layers.append(layer)

    def fit(self,
            X: np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame,
            y: Optional[np.ndarray | pd.Series |pd.DataFrame] = None,
            *, epochs: int=1,
            generator: Optional[torch.Generator] = None,
            shuffle: bool = False,
            batch_size: Optional[int]=0,
            validation_split: Optional[float] = None,
            validation_data: Optional[List | Tuple | DataLoader |
                                    Dataset | TensorDataset] = None,
            verbose: int = 1,
            **kwargs):
        
        from exttorch.history import History
        from exttorch.__data_handle import DataHandler

        # Initializer the History object
        history = History(self.metrics)

        if validation_split is not None and validation_data is None:
            for epoch in range(epochs):
                if verbose != 0:
                    # Print the epochs
                    print(f"Epoch {epoch + 1}/{epochs}")

                # Initializer the data
                data = DataHandler(X, y,
                                batch_size=batch_size if batch_size else None,
                                shuffle=shuffle,
                                generator=generator,
                                **kwargs)

                # Get the train and validation sample
                train_sample, val_sample = data(validation_split)

                # Train the train sample
                train_metric = self.__train(
                            train_sample,
                            y = None,
                            batch_size = batch_size if batch_size else None,
                            shuffle = shuffle,
                            generator = generator,
                            verbose=verbose,
                            **kwargs
                        )



                # Add the train metric to the history
                history.add_history(train_metric)


                # Evaluate the validation sample
                val_metric = self.evaluate(
                            val_sample,
                            y = None,
                            batch_size = batch_size if batch_size else None,
                            shuffle = shuffle,
                            generator = generator,
                            verbose=verbose,
                            **kwargs
                        )


                # Make a copy from train_metric dictionary.
                metrics = train_metric.copy()

                # Update the metrics by adding val_metric.
                metrics.update(val_metric)

                # Show the progress bar on each epoch
                self.__progbar.add(1, metrics.items())

                # Add the validation metric to the history
                history.add_history(val_metric)

        elif validation_data is not None:
            for epoch in range(epochs):
                if verbose != 0:
                    # Print the epochs
                    print(f"Epoch {epoch + 1}/{epochs}")

                # Initializer the data
                train_data = DataHandler(X, y,
                                batch_size=batch_size if batch_size else None,
                                shuffle=shuffle,
                                generator=generator,
                                **kwargs)()

                # Train the train sample
                train_metric = self.__train(
                            train_data,
                            y = None,
                            batch_size = batch_size if batch_size else None,
                            shuffle = shuffle,
                            generator = generator,
                            verbose=verbose,
                            **kwargs
                        )


                if ((isinstance(validation_data, list) or
                    isinstance(validation_data, tuple))
                and len(validation_data) == 2):
                    # Initializer the data
                    val_sample = DataHandler(
                        validation_data[0], validation_data[1],
                        batch_size=batch_size if batch_size else None,
                        shuffle=shuffle,
                        generator=generator,
                        **kwargs)()
                else:
                    # Initializer the data
                    val_sample = DataHandler(
                        validation_data, y=None,
                        batch_size=batch_size if batch_size else None,
                        shuffle=shuffle,
                        generator=generator,
                        **kwargs)()

                # Add the train metric to the history
                history.add_history(train_metric)


                # Evaluate the validation sample
                val_metric = self.evaluate(
                            val_sample,
                            y = None,
                            batch_size = batch_size if batch_size else None,
                            shuffle = shuffle,
                            generator = generator,
                            verbose=verbose,
                            **kwargs
                        )


                # Make a copy from train_metric dictionary.
                metrics = train_metric.copy()

                # Update the metrics by adding val_metric.
                metrics.update(val_metric)

                # Show the progress bar on each epoch
                self.__progbar.add(1, metrics.items())

                # Add the validation metric to the history
                history.add_history(val_metric)

        else:
            for epoch in range(epochs):
                if verbose != 0:
                    # Print the epochs
                    print(f"Epoch {epoch + 1}/{epochs}")

                # Initializer the data
                data = DataHandler(X, y,
                                batch_size=batch_size if batch_size else None,
                                shuffle=shuffle,
                                generator=generator,
                                **kwargs)()

                # Train the full dataset
                train_metric = self.__train(
                                data,
                                y = None,
                                batch_size = batch_size,
                                shuffle = shuffle,
                                generator = generator,
                                verbose=verbose,
                                **kwargs
                            )

                # Show the progress bar on each epoch
                self.__progbar.add(1, train_metric.items())

                # Add the train metric to the history
                history.add_history(train_metric)



        return history

    def predict_proba(self, X):
        x = (X.double()
                if type(X) == torch.Tensor
                else torch.tensor(X).double()
                ).to(self.__device)
        # Make prediction and get probabilities
        proba = self.__model(x)
        return proba.cpu().detach().numpy()

    def predict(self, X):
        from src.exttorch.__data_handle import SinglePredictionsFormat
        
        # Get the probabilities of x
        proba = self.predict_proba(X)

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(proba)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        formatted_prediction = formatted_prediction.numpy().T
        return formatted_prediction[0] if len(formatted_prediction) == 1 else formatted_prediction

    def __handle_one_hot(self, target):
        from torch.nn import functional as f
        loss_class_names = ["BCELoss", "BCEWithLogitsLoss"]

        return (f.one_hot(target, num_classes=2).double()
                if type(self.loss).__name__ in loss_class_names
                else target)

    def __train(
            self,
            X: np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame,
            y: Optional[np.ndarray | pd.Series |pd.DataFrame] = None,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
            generator: Optional[torch.Generator] = None,
            verbose: int = 1,
            **kwargs
            ) -> Dict:
        if self.optimizer is None or self.loss is None:
            raise TypeError("Compile the model with `model.compile` before " +
                            "fitting the model")

        metric_storage = None

        # Initializer the loss storage
        loss_storage = LossStorage()

        if self.metrics:
            # Create the list for metric
            metric_storage = MetricStorage(
                self.metrics,
                batch_size=batch_size)


        # Indicate the model to train
        self.__model.train()

        # Initializer the data
        data = DataHandler(X, y,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            generator=generator,
                            **kwargs)()

        # Get the data size
        self.__data_size = len(data)

        # Instantiate the progress bar
        progbar = Progbar(len(data), verbose=verbose)

        # Assign progbar
        self.__progbar = progbar


        # Loop over the data
        for idx, (feature, label) in enumerate(data):
            # Zero the gradient.
            self.optimizer.zero_grad()

            # Set the device for X and y
            feature, label = (feature.to(self.__device),
                        label.to(self.__device))

            # Make prediction
            predict = self.__model(feature.double())

            # Check if using BCELoss optimizer
            target = self.__handle_one_hot(label)


            # Change size of torch.size([1]) to torch.size([1, 1])
            target = (target.view(1, 1)
                    if (target.dim() == 1
                        and target.dtype in [torch.float32, torch.float64])
                    else target)

            # Compute the loss
            loss = self.loss(predict, target)

            # Add loss to the storage
            loss_storage.loss = loss.item()

            if idx != len(data) - 1:
                # Update the progress bar
                progbar.update(idx + 1, [("loss", loss_storage.loss)])


            if self.metrics and metric_storage:
                metric_storage.add_metric(predict, label=label)

            # Compute the gradient
            loss.backward()

            # update the parameters
            self.optimizer.step()

        if self.metrics and metric_storage:
            measurements = metric_storage.metrics(y=y)
            measurements['loss'] = loss_storage.loss

            # Place the val_loss to first position
            measurements = change_metric_first_position(measurements)

            return measurements

        return {'loss': loss_storage.loss}

    def evaluate(self,
            X: np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame,
            y: Optional[np.ndarray | pd.Series |pd.DataFrame] = None,
            batch_size: Optional[int] = 1,
            shuffle: bool = False,
            generator: Optional[torch.Generator] = None,
            verbose: int = 1,
            **kwargs):
        metric_storage = None

        # Initializer the loss storage
        loss_storage = LossStorage()

        if self.metrics:
            # Create the list for metric
            metric_storage = MetricStorage(
                self.metrics,
                batch_size=batch_size)


        # Indicate the model to train
        self.__model.eval()

        # Initializer the data
        data = DataHandler(X, y,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            generator=generator,
                            **kwargs)()

        # Instantiate the progress bar
        progbar = Progbar(len(data), verbose=verbose)

        with torch.no_grad():
            # Loop over the data
            for idx, (feature, label) in enumerate(data):

                # Set the device for X and y
                feature, label = (feature.to(self.__device),
                                label.to(self.__device))

                # Make prediction
                predict = self.__model(feature.double())

                if self.metrics and metric_storage:
                    metric_storage.add_metric(predict, label)

                # Check if using BCELoss optimizer
                target = self.__handle_one_hot(label)

                # Change size of torch.size([1]) to torch.size([1, 1])
                target = (target.view(1, 1)
                    if (target.dim() == 1 and
                        target.dtype in [torch.float32, torch.float64])
                    else target)

                if self.loss is not None:
                    # Compute the loss
                    loss = self.loss(predict, target)

                    # Add loss to the storage
                    loss_storage.loss = loss.item()

                    if idx != len(data) - 1:
                        # Update the progress bar
                        progbar.update(idx + 1, [("loss", loss_storage.loss)])



        if self.metrics and metric_storage:
            measurements = metric_storage.metrics(y)
            measurements['loss'] = loss_storage.loss

            # Place the val_loss to first position
            measurements = change_metric_first_position(measurements)

            # Add val to each key
            measurements = ({'val_' + key: value
                                for key, value in measurements.items()})

            return measurements
        return {'val_loss': loss_storage.loss}

    def compile(self,
                optimizer: Any,
                loss: Any,
                metrics: Optional[List[Metric|str]] = None,
                device: str = 'cpu'):
        self.__model = self.__model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.metrics: Optional[List] = (
            str_val_to_metric(metrics) if metrics is not None else [])
        self.__device = device
