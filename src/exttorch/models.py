"""Praise Ye The Lord Your God"""

# Import libraries
import torch as __torch__
from torch import nn as __nn__
from typing import Any as __Any__
from typing import List as __List__
from exttorch.losses import Loss as __Loss__
from exttorch._data_handle import DataHandler as __DataHandler__
from exttorch._metrics_handles import MetricStorage as __MetricStorage__
from exttorch.history import History as __History__ 
from exttorch.utils import ProgressBar as __ProgressBar__
from exttorch.losses import __change_str_to_loss as __change_str_to_loss__
from exttorch.optimizers import __change_str_to_optimizer as __change_str_to_optimizer__
from exttorch.metrics import Metric as __Metric__
from exttorch.optimizers import Optimizer as __Optimizer__
from exttorch.callbacks import Callback as __Callback__
from sklearn.utils.validation import check_is_fitted as __check_is_fitted__
from sklearn.base import (
    BaseEstimator as __BaseEstimator,
    TransformerMixin as __TransformerMixin,
)


class Sequential(__nn__.Module):
    def __init__(self, layers: list = None, device: str = "cpu") -> None:
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
        >>> from exttorch.models import Sequential
        >>> from torch import nn
        >>> from sklearn.datasets import load_iris
        >>>
        >>> # Load the iris dataset
        >>> x, y = load_iris(return_X_y=True)
        >>>
        >>> # Create the model
        >>> model = Sequential([
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
        super(Sequential, self).__init__()
        self.__xm = None

        match device:
            case "TPU" | "tpu":
                import torch_xla.core.xla_model as xm  # type: ignore

                self.__xm = xm
            case "GPU" | "gpu" | "cuda" | "CUDA":

                if __torch__.cuda.is_available():
                    device = device if device.startswith("cuda") else "cuda"
                    self.__device = __torch__.device(device)
                else:
                    raise ValueError("GPU is not available")
            case "CPU" | "cpu":
                self.__device = __torch__.device("cpu")
            case _:
                raise ValueError("device must be either 'TPU', 'GPU' or 'CPU'.")

        self.loss = None
        self.loss_obj = None
        self.optimizer = None
        self.optimizer_obj = None
        self.layers = layers if layers else []
        self.metrics = None
        self.__callbacks = None
        self.__progressbar = None
        self.stop_training = False
        self.__device = None
        self.__verbose = None
        self.__progressbar_color = None
        self.__progressbar_empty_color = None
        self.__progressbar_width = None
        self.__progressbar_dff_color = None
        self.__progressbar_style = None
        self.__progressbar_show_check_mark = None

    def get_weights(self):
        return self.__model.state_dict()

    def set_weights(self, weight):
        self.__model.load_state_dict(weight)

    def add(self, layer: __nn__.Module):
        self.layers.append(layer)

    def __handle_callbacks(self, callback_method, logs=None, epoch: int = None):

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
                        callback.on_epoch_begin(epoch)
                    case "on_epoch_end":
                        callback.on_epoch_end(epoch, logs)

    def fit(
        self,
        X,
        y=None,
        *,
        epochs: int = 1,
        random_seed=None,
        shuffle: bool = False,
        batch_size: int = 1,
        val_batch_size: int | None = None,
        validation_split: float = None,
        validation_data=None,
        verbose: str | None = "verbose",
        callbacks: __List__[__Callback__] = None,
        nprocs: int = 1,
        progressbar_width: int = 20,
        progressbar_dff_color: bool = False,
        progressbar_style: str = "default",
        progressbar_show_check_mark: bool = True,
        progressbar_color: str = "\033[92m",
        progressbar_empty_color: str = "\033[90m",
        **dataloader_kwargs,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
            X : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
                Training feature for training the model.
            y : (Optional[np.ndarray | pd.Series |pd.DataFrame]) None by default,
                Training label for training the model.
            epochs : (int) 1 by default,
                Number of epochs to train the model.
            generator : (Optional[torch.Generator]) None by default,
                For generator reproducibility of data.
            shuffle : (bool) False by default,
                Shuffle the data.
            batch_size : (Optional[int]) None by default,
                Batch size for training data.
            val_batch_size : (Optional[int]) batch_size by default
                Batch size for validation data
            validation_split : (Optional[float]) None by default,
                Split the dataset into train and validation data using
                validation_split as a test size and leaving the rest for
                train size.
            validation_data : (Optional[List | Tuple | DataLoader | Dataset | TensorDataset]) None by default,
                Data for validating model performance
            verbose : (str | None) verbose by default,
                Handles the model progress bar.
                If verbose is None, no progress bar is shown.
                Option :
                    - verbose: Displays the progress bar.
                    - silent: No progress bar is shown.
                    - silent_verbose: Displays the current batch, bar and elapsed time.
                    - silent_verbose_suffix: Displays the current batch and metrics.
                    - silent_epoch: Displays the current batch, bar, elapsed time and metrics but not epochs.
                    - silent_epoch_suffix: Displays the current batch and metrics but not epochs.
            callbacks: (Optional[List[Callback]])
                Model list of callbacks.
            nprocs: (int)
                The number of processes/devices for the replication.
                At the moment, if specified, can be either 1 or the maximum number of devices.
            random_seed: (int)
                Random seed for reproducibility.
            progressbar_width: (int)
                The width of the progress bar.
            progressbar_dff_color: (bool)
                If True, the progress bar will be displayed in a different color.
            progressbar_style: (str) `default` by default
                The style of the progress bar. option:
                    - default: Default style.
                    - square: Square style.
                    - circle: Circle style.
            progressbar_show_check_mark: (bool)
                If True, a check mark will be shown at the end of the progress bar.
            progressbar_color: (str)
                The color of the progress bar.
            progressbar_empty_color: (str)
                The color of the empty part of the progress bar.
            dataloader_kwargs: (Optional[Dict])
                Additional arguments for DataLoader.
        """

        self.stop_training = False

        if (
            validation_data and len(validation_data) > 0
        ) or validation_split is not None:
            show_val_metrics = True
        else:
            show_val_metrics = False

        # Instantiate the progress bar
        self.__progressbar = __ProgressBar__(
            show_val_metrics=show_val_metrics,
            verbose=verbose,
            bar_width=progressbar_width,
            show_diff_color=progressbar_dff_color,
            style=progressbar_style,
            show_check_mark=progressbar_show_check_mark,
            progress_color=progressbar_color,
            empty_color=progressbar_empty_color,
        )

        self.__verbose = verbose
        self.__progressbar_width = progressbar_width
        self.__progressbar_dff_color = progressbar_dff_color
        self.__progressbar_style = progressbar_style
        self.__progressbar_show_check_mark = progressbar_show_check_mark
        self.__progressbar_color = progressbar_color
        self.__progressbar_empty_color = progressbar_empty_color

        # Set the val_batch_size to batch_size if None
        val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        # Initializer the History object
        history = __History__(self.metrics)

        if type(random_seed) == int:
            # Set the random seed
            __torch__.manual_seed(random_seed)

        if callbacks is not None:
            self.__callbacks = callbacks

        def training():

            # self.__model_list = _nn.ModuleList(self.layers).to(self.__device).float()

            # Initialize the model
            self.__model = __nn__.Sequential(*self.layers)

            if self.__xm is not None:
                self.__device = self.__xm.xla_device()

            # Instantiate the Loss and optimizer
            self.loss = self.loss_obj()
            self.optimizer = self.optimizer_obj(self.__model.parameters())
            self.__model = self.__model.to(self.__device)

            if validation_split is not None and validation_data is None:

                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")

                # Initializer the data
                data = __DataHandler__(
                    X,
                    y,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    shuffle=shuffle,
                    random_seed=random_seed,
                    device=self.__device,
                    **dataloader_kwargs,
                )

                # Get the train and validation sample
                train_sample, val_sample = data.data_preprocessing(
                    nprocs, val_size=validation_split
                )

                for epoch in range(epochs):

                    # Handle the callbacks on epoch begin
                    self.__handle_callbacks("on_epoch_begin", epoch=epoch)

                    if verbose and "epoch" not in verbose:
                        # Print the epochs
                        print(f"Epoch {epoch + 1}/{epochs}")

                    # Train the train sample
                    train_metric = self.__train(
                        train_sample,
                        y=None,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        verbose=verbose,
                        nprocs=nprocs,
                        show_val_metrics=True,
                        **dataloader_kwargs,
                    )

                    # Add the train metric to the history
                    history.add_history(train_metric)

                    # Evaluate the validation sample
                    val_metric = self.evaluate(
                        val_sample,
                        y=None,
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        verbose=None,
                        nprocs=nprocs,
                        **dataloader_kwargs,
                    )

                    # Add the validation metric to the history
                    history.add_history(val_metric)

                    # Make a copy
                    metric_copy = train_metric.copy()
                    metric_copy.update(val_metric)

                    # Handle the callbacks on epoch end
                    self.__handle_callbacks(
                        "on_epoch_end", logs=metric_copy, epoch=epoch
                    )

                    if self.stop_training:
                        break

                # Handle the callbacks on train end
                self.__handle_callbacks("on_train_end", logs=history.history)

            elif validation_data is not None:

                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")

                # Initializer the data
                train_data = __DataHandler__(
                    X,
                    y,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    shuffle=shuffle,
                    random_seed=random_seed,
                    device=self.__device,
                    **dataloader_kwargs,
                ).data_preprocessing(nprocs)

                if (
                    isinstance(validation_data, list)
                    or isinstance(validation_data, tuple)
                ) and len(validation_data) == 2:
                    # Initializer the data
                    val_sample = __DataHandler__(
                        validation_data[0],
                        validation_data[1],
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        device=self.__device,
                        **dataloader_kwargs,
                    ).data_preprocessing(nprocs)
                else:
                    # Initializer the data
                    val_sample = __DataHandler__(
                        validation_data,
                        y=None,
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        device=self.__device,
                        **dataloader_kwargs,
                    ).data_preprocessing(nprocs)

                for epoch in range(epochs):
                    # Handle the callbacks on epoch begin
                    self.__handle_callbacks("on_epoch_begin", epoch=epoch)

                    if verbose and "epoch" not in verbose:
                        # Print the epochs
                        print(f"Epoch {epoch + 1}/{epochs}")

                    # Train the train sample
                    train_metric = self.__train(
                        train_data,
                        y=None,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        verbose=verbose,
                        nprocs=nprocs,
                        show_val_metrics=True,
                        **dataloader_kwargs,
                    )

                    # Add the train metric to the history
                    history.add_history(train_metric)

                    # Evaluate the validation sample
                    val_metric = self.evaluate(
                        val_sample,
                        y=None,
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        verbose=None,
                        nprocs=nprocs,
                        **dataloader_kwargs,
                    )

                    # Add the validation metric to the history
                    history.add_history(val_metric)

                    # Make a copy
                    metric_copy = train_metric.copy()
                    metric_copy.update(val_metric)

                    # Handle the callbacks on epoch end
                    self.__handle_callbacks(
                        "on_epoch_end", logs=metric_copy, epoch=epoch
                    )

                    if self.stop_training:
                        break

                # Handle the callbacks on train end
                self.__handle_callbacks("on_train_end", logs=history.history)

            else:
                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")

                # Initializer the data
                data = __DataHandler__(
                    X,
                    y,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    shuffle=shuffle,
                    random_seed=random_seed,
                    device=self.__device,
                    **dataloader_kwargs,
                ).data_preprocessing(nprocs)

                for epoch in range(1, epochs + 1):
                    # Handle the callbacks on epoch begin
                    self.__handle_callbacks("on_epoch_begin", epoch=epoch)

                    if verbose and "epoch" not in verbose:
                        # Print the epochs
                        print(f"Epoch {epoch}/{epochs}")

                    # Train the full dataset
                    train_metric = self.__train(
                        data,
                        y=None,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        random_seed=random_seed,
                        verbose=verbose,
                        nprocs=nprocs,
                        **dataloader_kwargs,
                    )

                    # Add the train metric to the history
                    history.add_history(train_metric)

                    print(end="\n")

                    # Handle the callbacks on epoch end
                    self.__handle_callbacks(
                        "on_epoch_end", epoch=epoch, logs=train_metric
                    )

                    if self.stop_training:
                        break

                # Handle the callbacks on train end
                self.__handle_callbacks("on_train_end", logs=history.history)

        with self.__progressbar:
            training()

        # Set the show validation metrics to False
        self.__progressbar.show_val_metrics = False

        return history

    def predict_proba(self, X, verbose: str | None = "inherited"):
        import torch
        import numpy as np
        from .utils import ProgressBar
        from torch.nn import functional as f

        x = (X.double() if type(X) == torch.Tensor else torch.tensor(X).double()).to(
            self.__device
        )

        # Instantiate the progress bar
        progressbar = ProgressBar(
            show_val_metrics=False,
            verbose=self.__verbose if verbose == "inherited" else verbose,
            bar_width=self.__progressbar_width,
            show_diff_color=self.__progressbar_dff_color,
            style=self.__progressbar_style,
            show_check_mark=self.__progressbar_show_check_mark,
            progress_color=self.__progressbar_color,
            empty_color=self.__progressbar_empty_color,
            show_suffix=False,
        )
        # Set the progress bar total
        progressbar.total = len(x)

        # Empty list for probability
        probability = []

        with torch.no_grad():
            for i, data in enumerate(x):

                # Make prediction and get probabilities
                proba = self.__model(data.view(1, -1).float())

                # Append the probabilities to the list
                probability.append(proba.detach().reshape(1, -1).tolist()[0])

                # Update the progress bar
                progressbar.update(i + 1, [()])

        prob = f.softmax(torch.tensor(probability), dim=1)
        return prob

    def predict(self, X, verbose: str | None = "inherited"):
        from ._metrics_handles import SinglePredictionsFormat

        # Get the probabilities of x
        proba = self.predict_proba(X, verbose=verbose)

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(
            proba, self.__device, loss_name=type(self.loss).__name__
        )

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()
        formatted_prediction = formatted_prediction.T
        
        return (
            formatted_prediction[0]
            if len(formatted_prediction) == 1
            else formatted_prediction
        )

    def __handle_label(self, target):
        if self.loss.__class__.__name__ == "CrossEntropyLoss":
            return target.long()
        elif self.loss.__class__.__name__ == "NLLLoss":
            return target.long().flatten()
        return target.view(-1, 1)
    
    def __metrics_handler(
        self,
        metric_storage,
        predict,
        label,
        loss,
        ):
        """
        Handle the metrics for the model.
        Parameters
        ----------
            metric_storage : (MetricStorage)
                The metric storage object.
            predict : (torch.Tensor)
                The prediction of the model.
            label : (torch.Tensor)
                The label of the model.
            loss : (torch.Tensor)
                The loss of the model.
        """
        # Add the prediction, labels(target) and loss to metric storage
        metric_storage.add_model_results(
            predict.detach(),
            label=label.detach(),
            loss=loss.detach(),
        )

        # Measurement live update
        metric_storage.measurement_computation()
        
    def __train(
        self,
        X,
        y=None,
        batch_size: int = 1,
        shuffle: bool = False,
        random_seed=None,
        verbose: str | int | None = 1,
        nprocs: int = 1,
        show_val_metrics: bool = False,
        warmup_steps: bool = False,
        **kwargs,
    ) -> dict:
        """
        Trains the model.

        Parameters
        ----------
            X : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
                Training feature for training the model.
            y : (Optional[np.ndarray | pd.Series |pd.DataFrame])
                Training label for training the model.
            epochs : (int)
                Number of epochs to train the model.
            generator : (Optional[torch.Generator])
                For generator reproducibility of data.
            shuffle : (bool)
                Shuffle the data.
            batch_size : (Optional[int])
                Batch size for training the model.
            validation_split : (Optional[float])
                Split the dataset into train and validation data using
                validation_split as a test size and leaving the rest for
                train size.
            validation_data : (Optional[List | Tuple | DataLoader | Dataset | TensorDataset])
                Data for validating model performance
        """

        if self.optimizer is None or self.loss is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )

        metric_storage = None

        # Create the list for metric
        metric_storage = __MetricStorage__(
            self.__device,
            self.metrics,
            batch_size=batch_size,
            loss_name=type(self.loss).__name__,
        )

        # Indicate the model to train
        self.__model.train()
        
        # Initializer the data
        data = __DataHandler__(
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            device=self.__device,
            **kwargs,
        ).data_preprocessing(nprocs)

        # # Get the data size
        self.__train_data_size = len(data)

        # # Set the progress bar total
        self.__progressbar.total = len(data)

        # # Handle on batch begin callback
        self.__handle_callbacks("on_batch_begin")

        # Loop over the data
        for idx, (feature, label) in enumerate(data):

            feature, label = (
                feature.to(self.__device).float(),
                label.to(self.__device).float(),
            )

            # Zero the gradient.
            self.optimizer.zero_grad()

            # Make prediction
            predict = self.__model(feature).float()
                        
            # Changes data type or data shape
            label = self.__handle_label(label)

            # Compute the loss
            loss = self.loss(predict, label)
            
            # Handle the metrics
            self.__metrics_handler(
                metric_storage,
                predict,
                label,
                loss,
            )
                
            # Update the progress bar
            self.__progressbar.update(idx + 1, metric_storage.measurements.items())

            # Compute the gradient
            loss.backward()

            # update the parameters
            if self.__xm is not None:
                self.__xm.optimizer_step(self.optimizer)
                self.__xm.mark_step()
            else:
                self.optimizer.step()
        
        # Measurements
        measurements = metric_storage.measurements

        # Handle on batch begin callback
        self.__handle_callbacks("on_batch_end", logs=measurements)

        return measurements

    def evaluate(
        self,
        X,
        y=None,
        batch_size: int = 1,
        val_batch_size: int | None = None,
        shuffle: bool = False,
        random_seed: int | None = None,
        verbose: str | None = "inherited",
        nprocs: int = 1,
        **dataloader_kwargs,
    ):
        """
        Evaluate the model.

        Parameters
        ----------
            X : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
                Training feature for training the model.
            y : (Optional[np.ndarray | pd.Series |pd.DataFrame])
                Training label for training the model.
            generator : (Optional[torch.Generator])
                For generator reproducibility of data.
            shuffle : (bool)
                Shuffle the data.
            batch_size : (Optional[int])
                Batch size for training data.
            val_batch_size : (Optional[int])
                Batch size for validation data.
            verbose : (str | None) verbose by default,
                Handles the model progress bar.
                If verbose is None, no progress bar is shown.
                Option :
                    - verbose: Displays the progress bar.
                    - silent: No progress bar is shown.
                    - silent_verbose: Displays the current batch, bar and elapsed time.
                    - silent_verbose_suffix: Displays the current batch and metrics.
                    - silent_epoch: Displays the current batch, bar, elapsed time and metrics but not epochs.
                    - silent_epoch_suffix: Displays the current batch and metrics but not epochs.

            nprocs: (int)
                The number of processes/devices for the replication.
                At the moment, if specified, can be either 1 or the maximum number of devices.
            dataloader_kwargs: (Optional[Dict])
                Additional arguments for DataLoader.
        """

        metric_storage = None

        # Instantiate the progress bar
        eval_progressbar = __ProgressBar__(
            show_val_metrics=False,
            verbose=self.__verbose if verbose == "inherited" else verbose,
            bar_width=self.__progressbar_width,
            show_diff_color=self.__progressbar_dff_color,
            style=self.__progressbar_style,
            show_check_mark=self.__progressbar_show_check_mark,
            progress_color=self.__progressbar_color,
            empty_color=self.__progressbar_empty_color,
        )

        # Create the list for metric
        metric_storage = __MetricStorage__(
            self.__device,
            metrics_measures=self.metrics,
            batch_size=batch_size,
            train=False,
            loss_name=type(self.loss).__name__,
        )

        # Indicate the model to evaluate
        self.__model.eval()

        # Initializer the data
        data = __DataHandler__(
            X,
            y,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            device=self.__device,
            **dataloader_kwargs,
        ).data_preprocessing(nprocs=nprocs)

        # Set the progress bar total
        eval_progressbar.total = len(data)

        # Handle on validation begin
        self.__handle_callbacks("on_validation_begin")

        with __torch__.no_grad():
            # Loop over the data
            for idx, (feature, label) in enumerate(data):

                # Set the device for X and y
                feature, label = (
                    feature.to(self.__device).float(),
                    label.to(self.__device).float(),
                )

                # Make prediction
                predict = self.__model(feature)

                # Check if using BCELoss optimizer
                label = self.__handle_label(label)

                if self.loss is not None:
                    # Compute the loss
                    loss = self.loss(predict, label)

                # Handle the metrics
                self.__metrics_handler(
                    metric_storage,
                    predict,
                    label,
                    loss,
                )

                if self.__xm is not None:
                    self.__xm.mark_step()

                if not self.__progressbar.show_val_metrics and verbose is not None:
                    # Update the progress bar
                    eval_progressbar.update(
                        idx + 1, metric_storage.measurements.items()
                    )

        if self.__progressbar.show_val_metrics:
            self.__progressbar.last_update(metric_storage.measurements.items())

        # Final measurements
        measurements = metric_storage.measurements

        # Handle on validation end
        self.__handle_callbacks("on_validation_end", logs=measurements)

        return measurements

    def compile(
        self,
        optimizer: __Optimizer__ | str,
        loss: __Loss__ | str,
        metrics: __List__[str | __Metric__] | None = None,
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
            if isinstance(optimizer, __Optimizer__)
            else __change_str_to_optimizer__(optimizer)
        )
        self.loss_obj = (
            loss if isinstance(loss, __Loss__) else __change_str_to_loss__(loss)
        )
        self.metrics = __MetricStorage__.str_val_to_metric(metrics) if metrics is not None else []


class Wrapper(__BaseEstimator, __TransformerMixin):
    """
    Wrapper class for exttorch models to make them compatible with sklearn
    """

    def __init__(
        self,
        model: Sequential,
        loss: __Loss__,
        optimizer: __Optimizer__,
        metrics=None,
        **fit_kwargs,
    ):
        super().__init__()
        self.model = model
        self.fit_kwargs = fit_kwargs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.history = None

    def fit(self, X, y=None, **kwargs):
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics
        )
        self.history = self.model.fit(
            X, y, **self.fit_kwargs if len(self.fit_kwargs) > 0 else kwargs
        )
        self.is_fitted_ = True
        return self

    def predict(self, X, verbose: str | None = None):
        __check_is_fitted__(self, "is_fitted_")
        return self.model.predict(X, verbose=verbose)

    def score(self, X, y=None, verbose: str | None = None):
        __check_is_fitted__(self, "is_fitted_")
        return self.model.evaluate(X, y, verbose=verbose)
