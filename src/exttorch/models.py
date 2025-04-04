"""Praise Ye The Lord Your God"""

# Import libraries
from torch import nn as __nn__
from typing import Any as __Any__
from typing import List as __List__
from exttorch.losses import Loss as __Loss__
from exttorch.losses import __change_str_to_loss as __change_str_to_loss__
from exttorch.optimizers import __change_str_to_optimizer as __change_str_to_optimizer__
from exttorch.metrics import Metric as __Metric__
from exttorch.optimizers import Optimizer as __Optimizer__
from exttorch.callbacks import Callback as __Callback__
from exttorch._env import _ENV as __ENV__


class Sequential(__nn__.Module):
    def __init__(self, layers: list = None) -> None:
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

        self.loss = None
        self.optimizer = None
        self.layers = layers if layers else []
        self.metrics = None
        self.__callbacks = None
        self.__progbar = None
        self.stop_training = False
        self.__ENV = __ENV__

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
        verbose: str | int | None = 1,
        callbacks: __List__[__Callback__] = None,
        nprocs: int = 1,
        start_method="fork",
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
            verbose : (str | int) 1 by default,
                Handles the model progress bar.
            callbacks: (Optional[List[Callback]])
                Model list of callbacks.
            nprocs: (int)
                The number of processes/devices for the replication.
                At the moment, if specified, can be either 1 or the maximum number of devices.
            start_method: (str) "fork" by default
                The method to start the process.
            kwargs: (Optional[Dict])
                Additional arguments for DataLoader.
        """
        # Import libraries
        from .history import History
        from ._data_handle import DataHandler
        import torch

        self.stop_training = False

        # Set the val_batch_size to batch_size if None
        val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        # Initializer the History object
        history = History(self.metrics)

        if type(random_seed) == int:
            # Set the random seed
            torch.manual_seed(random_seed)

        if callbacks is not None:
            self.__callbacks = callbacks

        def training():

            self.__device = (
                self.__ENV["EXTTORCH_XM"].xla_device()
                if "EXTTORCH_TPU" in self.__ENV
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # self.__model_list = _nn.ModuleList(self.layers).to(self.__device).float()

            # Initialize the model
            self.__model = __nn__.Sequential(*self.layers).to(self.__device).float()

            # Instantiate the Loss and optimizer
            self.loss = self.loss()
            self.optimizer = self.optimizer(self.__model.parameters())

            if validation_split is not None and validation_data is None:

                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")
                
                # Initializer the data
                data = DataHandler(
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

                    if verbose != 0 and verbose is not None:
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
                        show_val_progress=True,
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

                    # Make a copy from train_metric dictionary.
                    metrics = train_metric.copy()

                    # Update the metrics by adding val_metric.
                    metrics.update(val_metric)

                    if verbose != 0 and verbose is not None:
                        # Show the progress bar on each epoch
                        self.__progbar.update(
                            self.__train_data_size, val_metric.items(), finalize=True
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

                    if "EXTTORCH_TPU" in self.__ENV:
                        self.__ENV["EXTTORCH_XM"].rendezvous("epoch_sync")

                # Handle the callbacks on train end
                self.__handle_callbacks("on_train_end", logs=history.history)

            elif validation_data is not None:

                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")
                
                # Initializer the data
                train_data = DataHandler(
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
                    val_sample = DataHandler(
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
                    val_sample = DataHandler(
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

                    if verbose != 0 and verbose is not None:
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
                        show_val_progress=True,
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

                    if verbose != 0 and verbose is not None:
                        # Show the progress bar on each epoch
                        self.__progbar.update(
                            self.__train_data_size, val_metric.items(), finalize=True
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

                    if "EXTTORCH_TPU" in self.__ENV:
                        self.__ENV["EXTTORCH_XM"].rendezvous("epoch_sync")

                # Handle the callbacks on train end
                self.__handle_callbacks("on_train_end", logs=history.history)

            else:
                # Handle the callbacks on train begin
                self.__handle_callbacks("on_train_begin")

                print(end="\n")

                # Initializer the data
                data = DataHandler(
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

                    if verbose != 0 and verbose is not None:
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

        training()
        
        return history

    def predict_proba(self, X):
        import torch

        x = (X.double() if type(X) == torch.Tensor else torch.tensor(X).double()).to(
            self.__device
        )
        # Make prediction and get probabilities
        proba = self.__model(x)
        return proba

    def predict(self, X):
        from ._data_handle import SinglePredictionsFormat

        # Get the probabilities of x
        proba = self.predict_proba(X)

        # Initializer the SinglePredictionsFormat object.
        single_format_prediction = SinglePredictionsFormat(proba)

        # Format the predictions.
        formatted_prediction = single_format_prediction.format_prediction()

        formatted_prediction = formatted_prediction.numpy().T
        return (
            formatted_prediction[0]
            if len(formatted_prediction) == 1
            else formatted_prediction
        )

    def __handle_one_hot(self, target):
        from torch.nn import functional as f

        loss_class_names = ["BCELoss", "BCEWithLogitsLoss"]
        target = target.long()
        
        return (
            f.one_hot(target, num_classes=2).float()
            if type(self.loss).__name__ in loss_class_names
            else target
        )

    def __train(
        self,
        X,
        y=None,
        batch_size: int = 1,
        shuffle: bool = False,
        random_seed=None,
        verbose: str | int | None = 1,
        nprocs: int = 1,
        show_val_progress: bool = False,
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
        # Import libraries
        import torch
        from IPython.display import clear_output

        if verbose:
            from keras.utils import Progbar  # type: ignore
        from ._metrics_handles import LossStorage, MetricStorage
        from ._metrics_handles import change_metric_first_position
        from ._data_handle import DataHandler

        if self.optimizer is None or self.loss is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )

        metric_storage = None

        # Create the list for metric
        metric_storage = MetricStorage(
            self.__device, self.metrics, batch_size=batch_size
        )

        # Indicate the model to train
        self.__model.train()

        # Initializer the data
        data = DataHandler(
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

        if verbose is not None:
            # Instantiate the progress bar
            self.__progbar = Progbar(len(data), verbose=verbose)

        # # Handle on batch begin callback
        self.__handle_callbacks("on_batch_begin")

        # Loop over the data
        for idx, (feature, label) in enumerate(data):

            feature, label = feature.to(self.__device), label.to(self.__device)
            
            # Zero the gradient.
            self.optimizer.zero_grad()

            # Make prediction
            predict = self.__model(feature.float())

            # Check if using BCELoss optimizer
            target = self.__handle_one_hot(label)

            # Compute the loss
            loss = self.loss(predict, target)
            
            # Add the prediction, labels(target) and loss to metric storage
            metric_storage.add_metric(predict, label=label, loss=loss.item())

            # Measurement live update
            measurements = metric_storage.measurements_compiler()

            # Compute the gradient
            loss.backward()

            # update the parameters
            if "EXTTORCH_TPU" in self.__ENV:
                self.__ENV["EXTTORCH_XM"].optimizer_step(self.optimizer)
                self.__ENV["EXTTORCH_XM"].mark_step()
            else:
                self.optimizer.step()

            if verbose is not None:
                if show_val_progress:
                    if idx < len(data) - 1:
                        self.__progbar.update(idx + 1, measurements)
                else:
                    # Update the progress bar
                    self.__progbar.update(idx + 1, measurements)
                    
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
        verbose: int | None = 1,
        nprocs: int = 1,
        **kwargs,
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
            verbose : (int | None)
                default 1, Displays the progress bar if 1 or None not to
                display progress bar.
        """
        # Import libraries
        import torch
        from ._metrics_handles import LossStorage, MetricStorage

        if verbose:
            from keras.utils import Progbar  # type: ignore
        from ._metrics_handles import change_metric_first_position
        from ._data_handle import DataHandler

        metric_storage = None

        # Create the list for metric
        metric_storage = MetricStorage(
            self.__device, metrics=self.metrics, batch_size=batch_size, train=False
        )

        # Indicate the model to evaluate
        self.__model.eval()

        # Initializer the data
        data = DataHandler(
            X,
            y,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            device=self.__device,
            **kwargs,
        ).data_preprocessing(nprocs=nprocs)

        # # Declare the progbar
        # progbar = None

        steps = 0

        if verbose:
            # Instantiate the progress bar
            progbar = Progbar(len(data), verbose=verbose)

        # Handle on validation begin
        self.__handle_callbacks("on_validation_begin")

        with torch.no_grad():
            # Loop over the data
            for idx, (feature, label) in enumerate(data):

                # Set the device for X and y
                feature, label = (feature.to(self.__device), label.to(self.__device))

                # Make prediction
                predict = self.__model(feature.float())

                # Check if using BCELoss optimizer
                target = self.__handle_one_hot(label)

                # # Change size of torch.size([1]) to torch.size([1, 1])
                # target = (
                #     target.view(1, 1)
                #     if (
                #         target.dim() == 1
                #         and target.dtype in [torch.float32, torch.float64]
                #     )
                #     else target
                # )

                if self.loss is not None:
                    # Compute the loss
                    loss = self.loss(predict, target)

                # Add the prediction, labels(target) and loss to metric storage
                metric_storage.add_metric(predict, label=label, loss=loss.item())

                # Measurement live update
                live_measurements = metric_storage.measurements_compiler()

                if verbose is not None:
                    if 0.0 not in list(dict(live_measurements).values()):
                        # Update the progress bar
                        progbar.update(idx + 1, live_measurements)

                if "EXTTORCH_TPU" in self.__ENV:
                    self.__ENV["EXTTORCH_XM"].mark_step()

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
        # Import libraries
        from ._metrics_handles import str_val_to_metric

        self.optimizer = (
            optimizer
            if isinstance(optimizer, __Optimizer__)
            else __change_str_to_optimizer__(optimizer)
        )
        self.loss = loss if isinstance(loss, __Loss__) else __change_str_to_loss__(loss)
        self.metrics = str_val_to_metric(metrics) if metrics is not None else []


if __name__ == "__main__":
    import doctest

    doctest.testmod()
