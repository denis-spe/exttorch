# Bless be the LORD GOD of Host.

# Import libraries.
import torch
import torch.nn as nn
from typing import List, Optional
from src.exttorch.history import History
import torch
import torch.nn as nn
from src.exttorch.metrics import Metric
from src.exttorch.__metrics_handles import MetricStorage
from src.exttorch.__data_handle import DataHandler
from src.exttorch.__types import (
    VerboseType, FillStyleType, EmptyStyleType, ProgressType, Weight
)
from src.exttorch.utils import ProgressBar

class ModelFit:
    # Constructor
    def __init__(self):
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
                    
    def __handle_label(self, target):
        if self.loss.__class__.__name__ == "CrossEntropyLoss":
            return target.long()
        elif self.loss.__class__.__name__ == "NLLLoss":
            return target.long().flatten()
        return target.view(-1, 1)

    # Public methods
    def fit(
        self,
        x,
        y=None,
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
        **dataloader_kwargs,
    ):

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
            raise ValueError("Provide either validation_data or validation_split, not both.")
        
        self.loss = self.loss_obj()
        self.optimizer = self.optimizer_obj(self.model.parameters())
        self.model = self.model.to(self._device)

        # Initialize the data handler
        data_handler = DataHandler(
            x=x,
            y=y,
            dataloader_kwargs=dict(
                batch_size=batch_size,
                shuffle=shuffle,
            ),
            val_dataloader_kwargs=dict(
                batch_size=val_batch_size,
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
                train=False
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
                logs=combined_metric if validation_data is not None or validation_split is not None else train_stage,
                epoch=epoch
            )
        
        # Handle on train end callback
        self.__handle_callbacks(
            "on_train_end",
            logs=combined_metric if validation_data is not None or validation_split is not None else train_stage
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

            # Changes data type or data shape
            label = self.__handle_label(label)
            
            # Compute the loss
            loss = self.loss(predict, label)

            # Update metric state
            self.train_metric_storage.update_state(
                predict,
                label=label,
                loss=loss,
            )
            
            # Update the progress bar
            self._progressbar.update(
                current_value=idx + 1,
                metrics=list(self.train_metric_storage.measurements.items())
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

    def val_stage(
        self, 
        val_data
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

        # Indicate the model to evaluate
        self.model.eval()

        with torch.no_grad():
            # Loop over the data
            for idx, (feature, label) in enumerate(val_data):
                
                # Handle on validation begin
                self.__handle_callbacks("on_validation_begin")

                # Set the device for X and y
                feature, label = (
                    feature.to(self._device).float(),
                    label.to(self._device).float(),
                )

                # Make prediction
                predict = self.model(feature).float()

                # Check if using BCELoss optimizer
                label = self.__handle_label(label)

                # Compute the loss
                loss = self.loss(predict, label)

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
 

class Model(ModelFit):
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