# Bless be the LORD GOD of Host.

from typing import List, Any, Dict, TypedDict

# Import libraries.
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from src.exttorch.__data_handle import DataHandler, ValidationData, Xdata, Ydata
from src.exttorch.__metrics_handles import MetricStorage
from src.exttorch.__types import (
    VerboseType,
    FillStyleType,
    EmptyStyleType,
    ProgressType,
    Weight,
)
from src.exttorch.history import History
from src.exttorch.losses import Loss
from src.exttorch.metrics import Metric
from src.exttorch.optimizers import Optimizer
from src.exttorch.utils import ProgressBar


class FitParameters(TypedDict, total=False):
    epochs: int
    random_seed: int | None
    shuffle: bool
    batch_size: int | None
    val_batch_size: int | None
    validation_split: float | None
    validation_data: ValidationData
    callbacks: List | None
    progress_bar_width: int
    progress_fill_style: FillStyleType
    progress_empty_style: EmptyStyleType
    progress_fill_color: str
    progress_empty_color: str
    progress_percentage_colors: List[str] | None
    progress_progress_type: ProgressType
    verbose: VerboseType


def random_state(seed: int | None):
    if isinstance(seed, int):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # If you use CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # If you want deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelFit:
    # Constructor
    def __init__(self):
        self.optimizer = None
        self.loss = None
        self.val_metric_storage = None
        self._progressbar: ProgressBar | None = None
        self.stop_training = False
        self._verbose: VerboseType = "full"
        self._progress_bar_width: int = 40
        self._progress_fill_style: FillStyleType = "━"
        self._progress_empty_style: EmptyStyleType = "━"
        self._progress_fill_color: str = "\033[92m"
        self._progress_empty_color: str = "\033[90m"
        self._progress_percentage_colors: List[str] | None = None
        self._progress_progress_type: ProgressType = "bar"
        self._val_data_size: int = 0
        self.metrics: List[Metric] = []
        self.layers: List[torch.nn.Module] = []
        self.optimizer_obj = None
        self.loss_obj = None
        self.model: nn.Module | None = None
        self._device: torch.device | str = torch.device("cpu")
        self.__callbacks: List | None = None

    # Private methods
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

    def loss_computation(
        self, 
        prediction: torch.Tensor, 
        label: torch.Tensor
        ) -> torch.Tensor:
        if self.loss.__class__.__name__ == "CrossEntropyLoss":
            label = label.long()
        elif self.loss.__class__.__name__ == "NLLLoss":
            label = label.long().flatten()

        if self.loss.__class__.__name__ in ['BCELoss', "MSELoss"]:
            prediction = prediction.view(*label.shape)
        
        return self.loss(prediction, label)

    # Public methods
    def fit(
        self,
            x: Xdata,
            y: Ydata = None,
        *,
        epochs: int = 1,
        random_seed: int | None = None,
        shuffle: bool = False,
        batch_size: int | None = 1,
        val_batch_size: int | None = 1,
        validation_split: float | None = None,
        validation_data=None,
        callbacks=None,
        progress_bar_width: int = 40,
        progress_fill_style: FillStyleType = "━",
        progress_empty_style: EmptyStyleType = "━",
        progress_fill_color: str = "\033[92m",
        progress_empty_color: str = "\033[90m",
        progress_percentage_colors=None,
        progress_progress_type: ProgressType = "bar",
        verbose: VerboseType = "full",
            val_dataloader_kwargs=None,
        **dataloader_kwargs,
    ):
        # Set the random seed
        random_state(random_seed)

        if val_dataloader_kwargs is None:
            val_dataloader_kwargs = {}

        # Stop training flag
        self.stop_training = False

        # Declare the train
        train_stage = None

        # Declare the combined_metric
        combined_metric = None

        # Instantiate the progress bar
        self._progressbar = ProgressBar(
            bar_width=progress_bar_width,
            fill_style=progress_fill_style,
            empty_style=progress_empty_style,
            fill_color=progress_fill_color,
            empty_color=progress_empty_color,
            percentage_colors=progress_percentage_colors,
            progress_type=progress_progress_type,
            verbose=verbose,
            epochs=epochs,
        )

        self._verbose = verbose
        self._progress_bar_width = progress_bar_width
        self._progress_fill_style: FillStyleType = progress_fill_style
        self._progress_empty_style: EmptyStyleType = progress_empty_style
        self._progress_fill_color = progress_fill_color
        self._progress_empty_color = progress_empty_color
        self._progress_percentage_colors = progress_percentage_colors
        self._progress_progress_type: ProgressType = progress_progress_type

        # Set the val_batch_size to batch_size if None
        val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        # Initializer the History object
        history = History(self.metrics)

        if type(random_seed) == int:
            # Set the random seed
            torch.manual_seed(random_seed)

        if callbacks is not None:
            self.__callbacks = callbacks

        # Initialize the model
        self.model = nn.Sequential(*self.layers)

        # Instantiate the Loss and optimizer
        if self.loss_obj is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )
        if self.optimizer_obj is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )

        if validation_data is not None and validation_split is not None:
            raise ValueError(
                "Provide either validation_data or validation_split, not both."
            )

        self.loss = self.loss_obj()
        self.optimizer = self.optimizer_obj(self.model.parameters())
        self.model = self.model.to(self._device)

        # Initialize the data handler
        data_handler = DataHandler(
            x=x,
            y=y,
            dataloader_kwargs=dict(
                batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs
            ),
            val_dataloader_kwargs=dict(
                batch_size=val_batch_size, **val_dataloader_kwargs
            ),
            validation_split=validation_split,
            validation_data=validation_data,
        )

        # Get the train and validation data
        train_data, val_data = data_handler.get_data()

        # Add the size of train
        self._progressbar.total = len(train_data)

        # Handle on train begin callback
        self.__handle_callbacks("on_train_begin")

        for epoch in range(epochs):
            # Handle on epoch begin callback
            self.__handle_callbacks("on_epoch_begin", epoch=epoch)

            # Set the epoch for progress bar
            self._progressbar.set_epoch(epoch)

            if self.stop_training:
                break

            # Train metrics storage
            self.train_metric_storage = MetricStorage(
                self._device,
                self.metrics,
                batch_size=batch_size,
                loss_name=type(self.loss).__name__,
            )

            # Validation metrics storage
            self.val_metric_storage = MetricStorage(
                self._device,
                self.metrics,
                batch_size=val_batch_size,
                loss_name=type(self.loss).__name__,
                train=False,
            )

            # Train stage
            train_stage = self.train_stage(train_data)

            # Update the history
            history.add_history(train_stage)

            if validation_data is not None or validation_split is not None:
                val_stage = self.val_stage(val_data)
                combined_metric = {**train_stage, **val_stage}

                # validation data size
                val_data_size = len(val_data.dataset)

                # Update the history
                history.add_history(combined_metric)

                # Last progress update
                self._progressbar.last_update(
                    val_data_size, list(combined_metric.items())
                )

            # Handle on epoch end callback
            self.__handle_callbacks(
                "on_epoch_end",
                logs=(
                    combined_metric
                    if validation_data is not None or validation_split is not None
                    else train_stage
                ),
                epoch=epoch,
            )

        # Handle on train end callback
        self.__handle_callbacks(
            "on_train_end",
            logs=(
                combined_metric
                if validation_data is not None or validation_split is not None
                else train_stage
            ),
        )

        print("\n")

        return history

    def train_stage(
        self,
        train_data,
    ) -> dict:

        # Validate the parameters
        if self.optimizer is None or self.loss is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )
        if self._progressbar is None:
            raise ValueError("Progress bar is not initialized.")

        if self.model is None:
            raise ValueError("Model is not initialized.")

        # Indicate the model to train
        self.model.train()

        # Loop over the data
        for idx, (feature, label) in enumerate(train_data):

            # # Handle on batch begin callback
            self.__handle_callbacks("on_batch_begin")

            feature, label = (
                feature.to(self._device).float(),
                label.to(self._device).float(),
            )

            # Zero the gradient.
            self.optimizer.zero_grad()

            # Make prediction
            predict = self.model(feature).float()

            # Compute the loss
            loss = self.loss_computation(predict, label)

            # Update metric state
            self.train_metric_storage.update_state(
                predict,
                label=label,
                loss=loss,
            )

            # Update the progress bar
            self._progressbar.update(
                current_value=idx + 1,
                metrics=list(self.train_metric_storage.measurements.items()),
            )

            # Compute the gradient
            loss.backward()

            # update the parameters
            self.optimizer.step()

            # Handle on batch begin callback
            self.__handle_callbacks(
                "on_batch_end", logs=self.train_metric_storage.measurements
            )

        return self.train_metric_storage.measurements

    def val_stage(self, val_data) -> dict:

        # Validate the parameters
        if self.loss is None:
            raise TypeError(
                "Compile the model with `model.compile` before " + "fitting the model"
            )

        if self.model is None:
            raise ValueError("Model is not initialized.")

        # Indicate the model to evaluate
        self.model.eval()

        with torch.no_grad():
            # Loop over the data
            for feature, label in val_data:

                # Handle on validation begin
                self.__handle_callbacks("on_validation_begin")

                # Set the device for X and y
                feature, label = (
                    feature.to(self._device).float(),
                    label.to(self._device).float(),
                )

                # Make prediction
                predict = self.model(feature).float()

                # Compute the loss
                loss = self.loss_computation(predict, label)

                # Update metric state
                self.val_metric_storage.update_state(
                    predict,
                    label=label,
                    loss=loss,
                )

                # Handle on validation end
                self.__handle_callbacks(
                    "on_validation_end", logs=self.val_metric_storage.measurements
                )

        return self.val_metric_storage.measurements

    def evaluate(
        self,
        x,
        y=None,
        batch_size: int | None = 1,
        progress_bar_width: int = 40,
        progress_fill_style: FillStyleType = "━",
        progress_empty_style: EmptyStyleType = "━",
        progress_fill_color: str = "\033[92m",
        progress_empty_color: str = "\033[90m",
        progress_percentage_colors=None,
        progress_progress_type: ProgressType = "bar",
        verbose: VerboseType = "full",
        **dataloader_kwargs,
    ) -> Dict:

        # Instantiate the progress bar
        progressbar = ProgressBar(
            bar_width=progress_bar_width,
            fill_style=progress_fill_style,
            empty_style=progress_empty_style,
            fill_color=progress_fill_color,
            empty_color=progress_empty_color,
            percentage_colors=progress_percentage_colors,
            progress_type=progress_progress_type,
            verbose=verbose,
            epochs=2,
        )

        # Declare and initializer DataHandler
        data = DataHandler(
            x=x,
            y=y,
            validation_data=None,
            validation_split=None,
            dataloader_kwargs=dict(batch_size=batch_size, **dataloader_kwargs),
            val_dataloader_kwargs={},
        ).get_data()[0]

        # Add the size of train
        progressbar.total = len(data)

        # Initializer the MetricStorage
        val_metric_storage = MetricStorage(
            self._device,
            self.metrics,
            batch_size=batch_size,
            loss_name=type(self.loss).__name__,
            train=False,
        )

        if self.model is None:
            raise ValueError("Model is not initialized.")

        # Indicate the model to evaluate
        self.model.eval()

        with torch.no_grad():
            # Loop over the data
            for idx, (feature, label) in enumerate(data):

                # Set the device for X and y
                feature, label = (
                    feature.to(self._device).float(),
                    label.to(self._device).float(),
                )

                # Make prediction
                predict = self.model(feature).float()

                # Compute the loss
                loss = self.loss_computation(predict, label)

                # Update metric state
                val_metric_storage.update_state(
                    predict,
                    label=label,
                    loss=loss,
                )

                # Update the progress bar
                progressbar.update(
                    idx + 1, metrics=list(val_metric_storage.measurements.items())
                )

        return val_metric_storage.measurements


class ModelCompilation:
    def compile(
        self,
        optimizer: Optimizer | str,
        loss: Loss | str,
        metrics: List[str | Metric] | None = None,
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

    @staticmethod
    def __str_val_to_metric__(
        metric_list: List[Any],
    ) -> List[Metric]:
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

        new_metric_list: List[Metric] = []
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


class ModelPrediction:
    def __init__(self):
        self._device: torch.device | str = torch.device("cpu")
        self.loss_obj = None
        self.model: nn.Module | None = None
        self._progressbar: ProgressBar | None = None
        self._verbose: VerboseType = "full"
        self._progress_bar_width: int = 40
        self._progress_fill_style: FillStyleType = "━"
        self._progress_empty_style: EmptyStyleType = "━"
        self._progress_fill_color: str = "\033[92m"
        self._progress_empty_color: str = "\033[90m"
        self._progress_percentage_colors: List[str] | None = None
        self._progress_progress_type: ProgressType = "bar"

    # Public methods
    def predict_proba(self, x, verbose=None):
        x = (x.float() if type(x) == torch.Tensor else torch.tensor(x).float()).to(
            self._device
        )

        if self.model is None:
            raise ValueError(
                "The model is not trained yet. Please train the model first."
            )

        # Instantiate the progress bar
        progressbar = ProgressBar(
            bar_width=self._progress_bar_width,
            fill_style=self._progress_fill_style,
            empty_style=self._progress_empty_style,
            fill_color=self._progress_fill_color,
            empty_color=self._progress_empty_color,
            percentage_colors=self._progress_percentage_colors,
            progress_type=self._progress_progress_type,
            verbose=self._verbose if verbose is None else verbose,
        )
        # Set the progress bar total
        progressbar.total = len(x)

        # Empty list for probability
        probability = []

        with torch.no_grad():
            for i, data in enumerate(x):

                # Make prediction and get probabilities
                proba = self.model(data)

                # Append the probabilities to the list
                probability.append(proba.detach().reshape(1, -1).tolist()[0])

                # Update the progress bar
                progressbar.update(i + 1)

        if type(self.loss_obj).__name__ == "CrossEntropyLoss":
            probability = softmax(torch.tensor(probability), dim=1).numpy()
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


class Model(ModelFit, ModelCompilation, ModelPrediction):
    def __init__(
        self,
    ):
        super().__init__()

    def model_state_dict(self) -> Weight:
        if self.model is not None:
            return self.model.state_dict()
        else:
            raise TypeError(
                "The model must be fitted before calling the get_weights method"
            )

    def load_model_state_dict(self, weight: Weight):
        if self.model is not None:
            self.model.load_state_dict(weight)
        else:
            raise TypeError(
                "The model must be fitted before calling the set_weights method"
            )

    def save(self, filepath: str):
        """
        Saves the model to the specified file path.
        Args:
            filepath (str): The path where the model will be saved.
        """
        import pickle, os

        if not os.path.exists(os.path.dirname(filepath)):
            print(f"Creating directory: {os.path.dirname(filepath)}")
            os.makedirs(os.path.dirname(filepath))

        if filepath.endswith(".ext"):
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

        elif filepath.endswith(".we"):
            weights = self.model_state_dict()
            torch.save(weights, filepath)
        else:
            raise ValueError("Filepath must end with .ext or .we")
