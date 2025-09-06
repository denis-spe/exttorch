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
from src.exttorch.utils import ModelFit
from src.exttorch import __types as __types__
from src.exttorch.__data_handle import DataHandler as __DataHandler__
from src.exttorch.__metrics_handles import MetricStorage as __MetricStorage__
from src.exttorch.__model import ModelModule as __ModelModule__
from src.exttorch.history import History as __History__
from src.exttorch.losses import Loss as __Loss__
from src.exttorch.metrics import Metric as __Metric__
from src.exttorch.optimizers import Optimizer as __Optimizer__
from src.exttorch.utils import ProgressBar as __ProgressBar__


class Stack(ModelFit):
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
        self.layers = []

    def add(self, layer: __types__.Layer):
        self.layers.append(layer)

    # def fit(
    #     self,
    #     x,
    #     y: __ArrayLike__ | None = None,
    #     *,
    #     epochs: int = 1,
    #     random_seed: int | None = None,
    #     shuffle: bool = False,
    #     batch_size: int = 1,
    #     val_batch_size: int | None = None,
    #     validation_split: float | None = None,
    #     validation_data=None,
    #     callbacks=None,
    #     nprocs: int = 1,
    #     progress_bar_width: int = 40,
    #     progress_fill_style: __types__.FillStyleType = "━",
    #     progress_empty_style: __types__.EmptyStyleType = "━",
    #     progress_fill_color: str = "\033[92m",
    #     progress_empty_color: str = "\033[90m",
    #     progress_percentage_colors: __tp__.List[str] | None = None,
    #     progress_progress_type: __types__.ProgressType = "bar",
    #     verbose: __tp__.Literal[
    #         None,
    #         0,
    #         1,
    #         2,
    #         "full",
    #         "hide-epoch",
    #         "hide-batch-size",
    #         "hide-metrics",
    #         "hide-train-metrics",
    #         "hide-val-metrics",
    #         "hide-progress-bar",
    #         "hide-time-estimation",
    #         "percentage",
    #         "only_percentage",
    #         "only_epochs",
    #         "only_batch_size",
    #         "only_metrics",
    #         "only_train_metrics",
    #         "only_val_metrics",
    #         "only_progress_bar",
    #         "only_time_estimation",
    #     ] = "full",
    #     **dataloader_kwargs: __tp__.Any,
    # ):

    #     self.stop_training = False

    #     # Instantiate the progress bar
    #     self.__progressbar = __ProgressBar__(
    #         bar_width=progress_bar_width,
    #         fill_style=progress_fill_style,
    #         empty_style=progress_empty_style,
    #         fill_color=progress_fill_color,
    #         empty_color=progress_empty_color,
    #         percentage_colors=progress_percentage_colors,
    #         progress_type=progress_progress_type,
    #         verbose=verbose,
    #         epochs=epochs,
    #     )

    #     self.__verbose = verbose
    #     self.__progress_bar_width = progress_bar_width
    #     self.__progress_fill_style: __types__.FillStyleType = progress_fill_style
    #     self.__progress_empty_style: __types__.EmptyStyleType = progress_empty_style
    #     self.__progress_fill_color = progress_fill_color
    #     self.__progress_empty_color = progress_empty_color
    #     self.__progress_percentage_colors = progress_percentage_colors
    #     self.__progress_progress_type: __types__.ProgressType = progress_progress_type

    #     # Set the val_batch_size to batch_size if None
    #     val_batch_size = val_batch_size if val_batch_size is not None else batch_size

    #     # Initializer the History object
    #     history = __History__(self.metrics)

    #     if type(random_seed) == int:
    #         # Set the random seed
    #         __torch__.manual_seed(random_seed)

    #     if callbacks is not None:
    #         self.__callbacks = callbacks

    #     def training():

    #         # self.__model_list = _nn.ModuleList(self.layers).to(self.__device).float()

    #         # Initialize the model
    #         self._model = __nn__.Sequential(*self.layers)

    #         # if self.__xm is not None:
    #         #     self.__device = self.__xm.xla_device()

    #         # Instantiate the Loss and optimizer
    #         if self.loss_obj is None:
    #             raise TypeError(
    #                 "Compile the model with `model.compile` before "
    #                 + "fitting the model"
    #             )
    #         if self.optimizer_obj is None:
    #             raise TypeError(
    #                 "Compile the model with `model.compile` before "
    #                 + "fitting the model"
    #             )
    #         self.loss = self.loss_obj()
    #         self.optimizer = self.optimizer_obj(self._model.parameters())
    #         self._model = self._model.to(self._device)

    #         if validation_split is not None and validation_data is None:

    #             # Handle the callbacks on train begin
    #             self._handle_callbacks("on_train_begin")

    #             # Initializer the data
    #             data = __DataHandler__(
    #                 x=x,
    #                 y=y,
    #                 batch_size=batch_size,
    #                 val_batch_size=val_batch_size,
    #                 shuffle=shuffle,
    #                 random_seed=random_seed,
    #                 **dataloader_kwargs,
    #             )

    #             # Get the train and validation sample
    #             train_sample, val_sample = data(val_size=validation_split)

    #             for epoch in range(epochs):

    #                 # Handle the callbacks on epoch begin
    #                 self._handle_callbacks("on_epoch_begin", epoch=epoch)

    #                 # Set the epoch to progress bar.
    #                 self.__progressbar.set_epoch(epoch)

    #                 # Train the train sample
    #                 train_metric = self.__train(
    #                     train_sample,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     nprocs=nprocs,
    #                     **dataloader_kwargs,
    #                 )

    #                 # Add the train metric to the history
    #                 history.add_history(train_metric)

    #                 # Evaluate the validation sample
    #                 val_metric = self.evaluate(
    #                     val_sample,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     val_batch_size=val_batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     verbose=None,
    #                     nprocs=nprocs,
    #                     **dataloader_kwargs,
    #                 )

    #                 # Add the validation metric to the history
    #                 history.add_history(val_metric)

    #                 # Make a copy
    #                 metric_copy = train_metric.copy()
    #                 metric_copy.update(val_metric)

    #                 # Last progress update
    #                 self.__progressbar.last_update(
    #                     self.__val_data_size, list(metric_copy.items())
    #                 )

    #                 # Handle the callbacks on epoch end
    #                 self._handle_callbacks(
    #                     "on_epoch_end", logs=metric_copy, epoch=epoch
    #                 )

    #                 if self.stop_training:
    #                     break

    #             # Handle the callbacks on train end
    #             self._handle_callbacks("on_train_end", logs=history.history)

    #         elif validation_data is not None:

    #             # Handle the callbacks on train begin
    #             self._handle_callbacks("on_train_begin")

    #             # Initializer the data
    #             train_data = __DataHandler__(
    #                 x,
    #                 y,
    #                 batch_size=batch_size,
    #                 val_batch_size=val_batch_size,
    #                 shuffle=shuffle,
    #                 random_seed=random_seed,
    #                 **dataloader_kwargs,
    #             )()

    #             if (
    #                 isinstance(validation_data, list)
    #                 or isinstance(validation_data, tuple)
    #             ) and len(validation_data) == 2:
    #                 # Initializer the data
    #                 val_sample = __DataHandler__(
    #                     validation_data[0],
    #                     validation_data[1],
    #                     batch_size=batch_size,
    #                     val_batch_size=val_batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     **dataloader_kwargs,
    #                 )
    #             else:
    #                 # Initializer the data
    #                 val_sample = __DataHandler__(
    #                     x=validation_data,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     val_batch_size=val_batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     **dataloader_kwargs,
    #                 )()

    #             for epoch in range(epochs):
    #                 # Handle the callbacks on epoch begin
    #                 self._handle_callbacks("on_epoch_begin", epoch=epoch)

    #                 # Set the epoch to progress bar.
    #                 self.__progressbar.set_epoch(epoch)

    #                 # Train the train sample
    #                 train_metric = self.__train(
    #                     train_data,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     nprocs=nprocs,
    #                     **dataloader_kwargs,
    #                 )

    #                 # Add the train metric to the history
    #                 history.add_history(train_metric)

    #                 # Evaluate the validation sample
    #                 val_metric = self.evaluate(
    #                     val_sample,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     val_batch_size=val_batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     verbose=None,
    #                     nprocs=nprocs,
    #                     **dataloader_kwargs,
    #                 )

    #                 # Add the validation metric to the history
    #                 history.add_history(val_metric)

    #                 # # Make a copy
    #                 metric_copy = train_metric.copy()
    #                 metric_copy.update(val_metric)

    #                 # # Last progress update
    #                 self.__progressbar.last_update(
    #                     self.__val_data_size, list(metric_copy.items())
    #                 )

    #                 # Handle the callbacks on epoch end
    #                 self._handle_callbacks(
    #                     "on_epoch_end", logs=metric_copy, epoch=epoch
    #                 )

    #                 if self.stop_training:
    #                     break

    #             # Handle the callbacks on train end
    #             self._handle_callbacks("on_train_end", logs=history.history)

    #         else:
    #             # Handle the callbacks on train begin
    #             self._handle_callbacks("on_train_begin")

    #             # Initializer the data
    #             data = __DataHandler__(
    #                 x,
    #                 y,
    #                 batch_size=batch_size,
    #                 val_batch_size=val_batch_size,
    #                 shuffle=shuffle,
    #                 random_seed=random_seed,
    #                 **dataloader_kwargs,
    #             )()

    #             for epoch in range(epochs):
    #                 # Handle the callbacks on epoch begin
    #                 self._handle_callbacks("on_epoch_begin", epoch=epoch)

    #                 # Set the epoch to progress bar.
    #                 self.__progressbar.set_epoch(epoch)

    #                 # Train the full dataset
    #                 train_metric = self.__train(
    #                     data,
    #                     y=None,
    #                     batch_size=batch_size,
    #                     shuffle=shuffle,
    #                     random_seed=random_seed,
    #                     nprocs=nprocs,
    #                     **dataloader_kwargs,
    #                 )

    #                 # Add the train metric to the history
    #                 history.add_history(train_metric)

    #                 # Handle the callbacks on epoch end
    #                 self._handle_callbacks(
    #                     "on_epoch_end", epoch=epoch, logs=train_metric
    #                 )

    #                 if self.stop_training:
    #                     break

    #             # Handle the callbacks on train end
    #             self._handle_callbacks("on_train_end", logs=history.history)

    #     training()
    #     print("\n")

    #     return history

    def predict_proba(self, x, verbose=None):
        import torch
        import numpy as np
        from .utils import ProgressBar
        from torch.nn import functional as f

        x = (x.float() if type(x) == torch.Tensor else torch.tensor(x).float()).to(
            self._device
        )

        # Instantiate the progress bar
        progressbar = ProgressBar(
            bar_width=self.__progress_bar_width,
            fill_style=self.__progress_fill_style,
            empty_style=self.__progress_empty_style,
            fill_color=self.__progress_fill_color,
            empty_color=self.__progress_empty_color,
            percentage_colors=self.__progress_percentage_colors,
            progress_type=self.__progress_progress_type,
            verbose=self.__verbose if verbose is None else verbose,
        )
        # Set the progress bar total
        progressbar.total = len(x)

        # Empty list for probability
        probability = []

        with torch.no_grad():
            for i, data in enumerate(x):

                # Make prediction and get probabilities
                proba = self._model(data)

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

    # def __handle_label(self, target):
    #     if self.loss.__class__.__name__ == "CrossEntropyLoss":
    #         return target.long()
    #     elif self.loss.__class__.__name__ == "NLLLoss":
    #         return target.long().flatten()
    #     return target.view(-1, 1)

    # @staticmethod
    # def __metrics_handler(
    #     metric_storage,
    #     predict,
    #     label,
    #     loss,
    # ):
    #     """
    #     Handle the metrics for the model.
    #     Parameters
    #     ----------
    #         metric_storage : (MetricStorage)
    #             The metric storage object.
    #         predict : (torch.Tensor)
    #             The prediction of the model.
    #         label : (torch.Tensor)
    #             The label of the model.
    #         loss : (torch.Tensor)
    #             The loss of the model.
    #     """
    #     # Add the prediction, labels(target) and loss to metric storage
    #     metric_storage.add_model_results(
    #         predict.detach(),
    #         label=label.detach(),
    #         loss=loss.detach(),
    #     )

    #     # Measurement live update
    #     metric_storage.measurement_computation()

    # def train_stage(
    #     self,
    #     train_data,
    #     y=None,
    #     batch_size: int = 1,
    #     shuffle: bool = False,
    #     random_seed=None,
    #     **kwargs,
    # ) -> dict:
    #     """
    #     Trains the model.

    #     Parameters
    #     ----------
    #         x : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
    #             Training feature for training the model.
    #         y : (Optional[np.ndarray | pd.Series |pd.DataFrame])
    #             Training label for training the model.
    #         epochs : (int)
    #             Number of epochs to train the model.
    #         generator : (Optional[torch.Generator])
    #             For generator reproducibility of data.
    #         shuffle : (bool)
    #             Shuffle the data.
    #         batch_size : (Optional[int])
    #             Batch size for training the model.
    #         validation_split : (Optional[float])
    #             Split the dataset into train and validation data using
    #             validation_split as a test size and leaving the rest for
    #             train size.
    #         validation_data : (Optional[List | Tuple | DataLoader | Dataset | TensorDataset])
    #             Data for validating model performance
    #     """

    #     if self.optimizer is None or self.loss is None:
    #         raise TypeError(
    #             "Compile the model with `model.compile` before " + "fitting the model"
    #         )

    #     # Create the list for metric
    #     metric_storage = __MetricStorage__(
    #         self._device,
    #         self.metrics,
    #         batch_size=batch_size,
    #         loss_name=type(self.loss).__name__,
    #     )

    #     # Indicate the model to train
    #     self._model.train()

    #     # Initializer the data
    #     data = __DataHandler__(
    #         x,
    #         y,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         random_seed=random_seed,
    #         **kwargs,
    #     )()

    #     # # Set the progress bar total
    #     self.__progressbar.total = len(data)

    #     # # Handle on batch begin callback
    #     self._handle_callbacks("on_batch_begin")

    #     # Loop over the data
    #     for idx, (feature, label) in enumerate(data):

    #         feature, label = (
    #             feature.to(self._device).float(),
    #             label.to(self._device).float(),
    #         )

    #         # Zero the gradient.
    #         self.optimizer.zero_grad()

    #         # Make prediction
    #         predict = self._model(feature).float()

    #         # Changes data type or data shape
    #         label = self.__handle_label(label)

    #         # Compute the loss
    #         loss = self.loss(predict, label)

    #         # Handle the metrics
    #         self.__metrics_handler(
    #             metric_storage,
    #             predict,
    #             label,
    #             loss,
    #         )

    #         # Update the progress bar
    #         self.__progressbar.update(
    #             idx + 1, list(metric_storage.measurements.items())
    #         )

    #         # Compute the gradient
    #         loss.backward()

    #         # update the parameters
    #         if self._xm is not None:
    #             self._xm.optimizer_step(self.optimizer)
    #             self._xm.mark_step()
    #         else:
    #             self.optimizer.step()

    #     # Measurements
    #     measurements = metric_storage.measurements

    #     # Handle on batch begin callback
    #     self._handle_callbacks("on_batch_end", logs=measurements)

    #     return measurements

    # def validation_stage(
    #     self,
    #     x,
    #     y=None,
    #     batch_size: int = 1,
    #     val_batch_size: int | None = None,
    #     shuffle: bool = False,
    #     random_seed: int | None = None,
    #     verbose=None,
    #     **dataloader_kwargs,
    # ):
    #     """
    #     Evaluate the model.

    #     Parameters
    #     ----------
    #         x : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
    #             Training feature for training the model.
    #         y : (Optional[np.ndarray | pd.Series |pd.DataFrame])
    #             Training label for training the model.
    #         shuffle : (bool)
    #             Shuffle the data.
    #         batch_size : (Optional[int])
    #             Batch size for training data.
    #         val_batch_size : (Optional[int])
    #             Batch size for validation data.
    #         verbose : (str | None) verbose by default,
    #             Handles the model progress bar.
    #             If verbose is None, no progress bar is shown.
    #             Option :
    #                 - verbose: Displays the progress bar.
    #                 - silent: No progress bar is shown.
    #                 - silent_verbose: Displays the current batch, bar and elapsed time.
    #                 - silent_verbose_suffix: Displays the current batch and metrics.
    #                 - silent_epoch: Displays the current batch, bar, elapsed time and metrics but not epochs.
    #                 - silent_epoch_suffix: Displays the current batch and metrics but not epochs.

    #         nprocs: (int)
    #             The number of processes/devices for the replication.
    #             At the moment, if specified, can be either 1 or the maximum number of devices.
    #         dataloader_kwargs: (Optional[Dict])
    #             Additional arguments for DataLoader.
    #     """

    #     # Instantiate the progress bar
    #     eval_progressbar = __ProgressBar__(
    #         bar_width=self.__progress_bar_width,
    #         fill_style=self.__progress_fill_style,
    #         empty_style=self.__progress_empty_style,
    #         fill_color=self.__progress_fill_color,
    #         empty_color=self.__progress_empty_color,
    #         percentage_colors=self.__progress_percentage_colors,
    #         progress_type=self.__progress_progress_type,
    #         verbose=self.__verbose if verbose is None else verbose,
    #     )

    #     # Create the list for metric
    #     metric_storage = __MetricStorage__(
    #         self._device,
    #         metrics_measures=self.metrics,
    #         batch_size=batch_size,
    #         train=False,
    #         loss_name=type(self.loss).__name__,
    #     )

    #     # Indicate the model to evaluate
    #     self._model.eval()

    #     # Initializer the data
    #     data = __DataHandler__(
    #         x,
    #         y,
    #         batch_size=batch_size,
    #         val_batch_size=val_batch_size,
    #         shuffle=shuffle,
    #         random_seed=random_seed,
    #         **dataloader_kwargs,
    #     )()

    #     # Set the progress bar total
    #     eval_progressbar.total = len(data)
    #     self.__val_data_size = len(data)

    #     # Handle on validation begin
    #     self._handle_callbacks("on_validation_begin")

    #     with __torch__.no_grad():
    #         # Loop over the data
    #         for idx, (feature, label) in enumerate(data):

    #             # Set the device for X and y
    #             feature, label = (
    #                 feature.to(self._device).float(),
    #                 label.to(self._device).float(),
    #             )

    #             # Make prediction
    #             predict = self._model(feature).float()

    #             # Check if using BCELoss optimizer
    #             label = self.__handle_label(label)

    #             if self.loss is not None:
    #                 # Compute the loss
    #                 loss = self.loss(predict, label)

    #             # Handle the metrics
    #             self.__metrics_handler(
    #                 metric_storage,
    #                 predict,
    #                 label,
    #                 loss,
    #             )

    #             if self._xm is not None:
    #                 self._xm.mark_step()

    #             if verbose is not None:
    #                 # Update the progress bar
    #                 eval_progressbar.update(
    #                     idx + 1, list(metric_storage.measurements.items())
    #                 )

    #     # Final measurements
    #     measurements = metric_storage.measurements

    #     # Handle on validation end
    #     self._handle_callbacks("on_validation_end", logs=measurements)

    #     return measurements

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

    def __handle_callbacks(
        self, callback_method: str, logs=None, epoch: int | None = None
    ):

        if self.__callbacks is not None:
            for callback in self.__callbacks:
                # Set the model and stop_training to the callback
                callback.model = self

                # Check if the present callback method
                match callback_method:
                    case "on_train_begin":
                        callback.on_train_begin()
                    case "on_train_end":
                        callback.on_train_end(logs)
                    case "on_validation_begin":
                        callback.on_validation_begin()
                    case "on_validation_end":
                        callback.on_validation_end(logs)
                    case "on_batch_begin":
                        callback.on_batch_begin()
                    case "on_batch_end":
                        callback.on_batch_end(logs)
                    case "on_epoch_begin":
                        if epoch is None:
                            raise ValueError(
                                "epoch must be provided for on_epoch_begin callback method"
                            )
                        callback.on_epoch_begin(epoch)
                    case "on_epoch_end":
                        if epoch is None:
                            raise ValueError(
                                "epoch must be provided for on_epoch_end callback method"
                            )
                        callback.on_epoch_end(epoch, logs)
                    case _:
                        raise ValueError(
                            "Unknown callback_method name: {}".format(callback_method)
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
        model: Stack,
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
