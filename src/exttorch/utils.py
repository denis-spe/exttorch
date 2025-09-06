# Holy, holy, holy is the LORD of host, the whole earth is full of his glory.

# Importing necessary modules

import sys
import time
from typing import List, Tuple
from dataclasses import dataclass
import itertools
from src.exttorch.history import History
import torch
import torch.nn as nn
from src.exttorch.metrics import Metric
from src.exttorch.__metrics_handles import MetricStorage
from src.exttorch.__data_handle import DataHandler
from src.exttorch.__types import FillStyleType, EmptyStyleType, ProgressType, VerboseType


@dataclass
class TimeEstimate:
    elapsed: int
    milliseconds: int


@dataclass
class BarStyle:
    fill_style: str
    empty_style: str


class ProgressBar:
    """
    A class represents a progress bar for tracking the progress of a model training, validation and predict.
    """

    # Constructor
    def __init__(
        self,
        bar_width: int = 40,
        epochs: int | None = None,
        fill_style: FillStyleType = "━",
        empty_style: EmptyStyleType = "━",
        fill_color: str = "\033[92m",  # Green
        empty_color: str = "\033[90m",  # Grey
        percentage_colors: List[str] | None = None,
        progress_type: ProgressType = "bar",
        verbose: VerboseType = "full",
    ):
        """
        Initializes the ProgressBar with total iterations, prefix and length of the bar.

        Args:
            bar_width (int): Width of the progress bar.
            epochs (int | None): Number of epochs. If None, no epoch information will be displayed.
            fill_style (str): Character used to fill the progress bar.
            empty_style (str): Character used to represent empty space in the progress bar.
            fill_color (str): Color code for the filled part of the progress bar.
            empty_color (str): Color code for the empty part of the progress bar.
            verbose (str | None): Verbosity level for displaying progress information.
                - None: No verbosity.
                - 0: Only current iteration and total iterations.
                - 1: Current iteration and total iterations with progress bar.
                - 2: Current iteration and total iterations with progress bar, elapsed time and milliseconds.
                - "full": Full verbosity with all information.
                - "hide-epoch": Hide epoch information.
                - "hide-batch-size": Hide batch size information.
                - "hide-metrics": Hide metrics information.
                - "hide-train-metrics": Hide training metrics information.
                - "hide-val-metrics": Hide validation metrics information.
                - "hide-progress-bar": Hide progress bar.
                - "hide-time-estimation": Hide time estimation.
                - "percentage": Show percentage completion.
                - "only_percentage": Show only percentage completion.
                - "only_epochs": Show only epoch information.
                - "only_batch_size": Show only batch size information.
                - "only_metrics": Show only metrics information.
                - "only_train_metrics": Show only training metrics information.
                - "only_val_metrics": Show only validation metrics information.
                - "only_progress_bar": Show only progress bar.
                - "only_time_estimation": Show only time estimation.
        """

        # Validate parameters
        self.param_validation(
            fill_style=fill_style,
            empty_style=empty_style,
            fill_color=fill_color,
            empty_color=empty_color,
            bar_width=bar_width,
            verbose=verbose,
            percentage_colors=percentage_colors,
            epochs=epochs,
        )

        # Initialize parameters
        self.__total = None
        self.__bar_width = bar_width
        self.__current_value = 0
        self.__epochs = epochs
        self.__start_time = time.time()
        self.__last_update_start_time = time.time()
        self.__last_time_estimate = None
        self.__fill_style = fill_style
        self.__empty_style = empty_style
        self.__fill_color = fill_color
        self.__empty_color = empty_color
        self.__reset_color = "\033[0m"  # Reset color
        self.__verbose = verbose
        self.__percentage_colors = percentage_colors
        self.__percent = 0
        self.__progress_type = progress_type
        self.__pre_epoch = None

        # Store original sequences
        self.__bounce_seq = [
            "▇",
            "▁",
            "▃",
            "▄",
            "▅",
            "▆",
            "▇",
            "█",
            "▇",
            "▆",
            "▅",
            "▄",
            "▃",
        ]
        self.__square_seq = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.__clock_seq = ["◴", "◷", "◶", "◵"]
        self.__arrows_seq = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
        self.__triangles_seq = ["◢", "◣", "◤", "◥"]
        self.__cross_seq = ["✶", "✸", "✹", "✺", "✻", "✼", "✽", "✾"]
        self.__moon_seq = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
        self.__pie_seq = ["◐", "◓", "◑", "◒"]

        # Initialize cycles
        self.__pie = itertools.cycle(self.__pie_seq)
        self.__square = itertools.cycle(self.__square_seq)
        self.__bounce = itertools.cycle(self.__bounce_seq)
        self.__clock = itertools.cycle(self.__clock_seq)
        self.__arrows = itertools.cycle(self.__arrows_seq)
        self.__triangles = itertools.cycle(self.__triangles_seq)
        self.__cross = itertools.cycle(self.__cross_seq)
        self.__moon = itertools.cycle(self.__moon_seq)

    @property
    def total(self):
        return self.__total

    @total.setter
    def total(self, value: int):
        if value <= 0:
            raise ValueError("Total must be a positive integer.")
        self.__total = value

    @property
    def __reset_cycles(self):
        """Reset all animation cycles to their first values"""
        self.__bounce = itertools.cycle(self.__bounce_seq)
        self.__square = itertools.cycle(self.__square_seq)
        self.__clock = itertools.cycle(self.__clock_seq)
        self.__arrows = itertools.cycle(self.__arrows_seq)
        self.__triangles = itertools.cycle(self.__triangles_seq)
        self.__cross = itertools.cycle(self.__cross_seq)
        self.__moon = itertools.cycle(self.__moon_seq)
        self.__pie = itertools.cycle(self.__pie_seq)

    def param_validation(
        self,
        fill_style: str,
        empty_style: str,
        fill_color: str,
        empty_color: str,
        bar_width: int,
        verbose: str | int | None,
        percentage_colors: List[str] | None = None,
        epochs: int | None = None,
    ):
        """
        Validates the parameters for the ProgressBar class.
        """
        symbols = [
            "━",
            "◉",
            "◆",
            "●",
            "█",
            "▮",
            "=",
            "#",
            "◎",
            "◇",
            "○",
            "░",
            "▯",
            "-",
            "▒",
            ".",
            "▶",
            "▷",
            "■",
            "□",
            "➤",
        ]
        if fill_style not in symbols:
            raise ValueError(f"Invalid fill_style. Choose from {symbols}.")
        if empty_style not in symbols:
            raise ValueError(f"Invalid empty_style. Choose from {symbols}.")
        verbose_lst = [
            None,
            0,
            1,
            2,
            "full",
            "hide-epoch",
            "hide-batch-size",
            "hide-metrics",
            "hide-train-metrics",
            "hide-val-metrics",
            "hide-progress-bar",
            "hide-time-estimation" "percentage",
            "only_percentage",
            "only_epochs",
            "only_batch_size",
            "only_metrics",
            "only_train_metrics",
            "only_val_metrics",
            "only_progress_bar",
            "only_time_estimation",
        ]
        if verbose not in verbose_lst:
            raise ValueError(f"Invalid verbose. Choose from {verbose_lst}.")
        if not fill_color.startswith("\033[") or not fill_color.endswith("m"):
            raise ValueError("Invalid fill_color. Must be a valid ANSI color code.")
        if not empty_color.startswith("\033[") or not empty_color.endswith("m"):
            raise ValueError("Invalid empty_color. Must be a valid ANSI color code.")
        if not isinstance(bar_width, int) or bar_width <= 0:
            raise ValueError("Invalid bar_width. Must be a positive integer.")
        if percentage_colors is not None and len(percentage_colors) != 4:
            raise ValueError(
                "Invalid percentage_colors. Must be a list of 4 color codes."
            )
        if epochs is not None and not isinstance(epochs, int) and epochs <= 0:
            raise ValueError("Invalid epochs. Must be a positive integer.")

    def set_epoch(self, epoch: int):
        """
        Sets the current epoch.

        Args:
            epoch (int): Current epoch number.
        """

        if self.__epochs is not None and epoch > self.__epochs:
            raise ValueError("Epoch exceeds the total number of epochs.")

        # Track epoch
        self.__pre_epoch = epoch + 1

        if (
            self.__epochs is not None
            and self.__verbose != "hide-epoch"
            and self.__verbose is not None
            and (
                isinstance(self.__verbose, str)
                and not self.__verbose.startswith("only")
            )
            and self.__verbose != 0
        ):
            print(f"\nEpoch {epoch + 1}/{self.__epochs}")
        elif self.__verbose == "hide-epoch" or (
            self.__verbose is not None
            and (
                isinstance(self.__verbose, str)
                and not self.__verbose.startswith("only")
            )
        ):
            print(f"\n")

    # Methods
    def __bar_style(self) -> BarStyle:
        """
        Returns the style of the progress bar.

        Returns:
            str: Style of the progress bar.
        """
        color = self.__fill_color

        if self.__percentage_colors is not None:
            if self.__percent <= 25:
                color = self.__percentage_colors[0]
            elif self.__percent <= 50:
                color = self.__percentage_colors[1]
            elif self.__percent <= 75:
                color = self.__percentage_colors[2]
            elif self.__percent <= 100:
                color = self.__percentage_colors[3]

        return BarStyle(
            fill_style=f"{color}{self.__fill_style}{self.__reset_color}",
            empty_style=f"{self.__empty_color}{self.__empty_style}{self.__reset_color}",
        )

    def __time_estimate(self, value: int, start_time: int) -> TimeEstimate:
        """
        Estimates the time remaining for the progress bar.

        Returns:
            TimeEstimate: Estimated time remaining in seconds and milliseconds.
        """
        now = time.time()
        elapsed = int(now - start_time)
        milliseconds = int((elapsed / value) * 1000) if value > 0 else 0
        return TimeEstimate(elapsed, milliseconds=milliseconds)

    def __selecting_verbose(
        self,
        bar: str,
        elapsed: str,
        milliseconds: str,
        formatted_metrics: str | None,
        percent: int,
    ):
        if formatted_metrics is not None:
            hide_train_metrics = formatted_metrics.split(" - ")
            hide_train_metrics = " - ".join(
                list(filter(lambda x: "val" not in x, hide_train_metrics))
            )
            hide_train_metrics = " - " + hide_train_metrics

            hide_val_metrics = formatted_metrics.split(" - ")
            hide_val_metrics = " - ".join(
                list(filter(lambda x: "val" in x, hide_val_metrics))
            )
            if self.__current_value == self.__total:
                hide_val_metrics = " - " + hide_val_metrics
        else:
            hide_train_metrics = ""
            hide_val_metrics = ""

        formatted_metrics = (
            f" - {formatted_metrics}" if formatted_metrics is not None else ""
        )

        verbose_style = {
            "full": f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}{formatted_metrics}",
            "hide-epoch": f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}{formatted_metrics}",
            "hide-batch-size": f"\r{bar} {elapsed} {milliseconds}{formatted_metrics}",
            "hide-metrics": f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}",
            "hide-train-metrics": f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}{hide_val_metrics}",
            "hide-val-metrics": f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}{hide_train_metrics}",
            "hide-progress-bar": f"\r{self.__current_value}/{self.__total} - {elapsed} {milliseconds}{formatted_metrics}",
            "hide-time-estimation": f"\r{self.__current_value}/{self.__total} {bar} {formatted_metrics}",
            "percentage": f"\r{percent}% {bar} {elapsed} {milliseconds}{formatted_metrics}",
            "only_percentage": f"\r{percent}%",
            "only_metrics": f"\r{formatted_metrics}",
            "only_train_metrics": f"\r{hide_train_metrics}",
            "only_val_metrics": f"\r{hide_val_metrics}",
            "only_progress_bar": f"\r{bar}",
            "only_time_estimation": f"\r{elapsed} {milliseconds}",
            "only_batch_size": f"\r{self.__current_value}/{self.__total}",
            1: f"\r{self.__current_value}/{self.__total}",
            2: f"\r{self.__current_value}/{self.__total} {bar}",
            3: f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}",
            4: f"\r{self.__current_value}/{self.__total} {bar} {elapsed} {milliseconds}{formatted_metrics}",
        }
        return verbose_style.get(self.__verbose)

    def __progress_symbol(
        self,
        progress_type: str,
        filled_length: int,
        empty_length: int,
        bar_style: BarStyle,
    ):
        match progress_type:
            case "bar":
                return (
                    f"{bar_style.fill_style}" * filled_length
                    + f"{bar_style.empty_style}" * empty_length
                )
            case "pie":
                next_pie = next(self.__pie)
                return f"{next_pie}"
            case "squares":
                next_square = next(self.__square)
                return f"{next_square}"
            case "bounce":
                next_bounce = next(self.__bounce)
                return f"{next_bounce}"
            case "clock":
                next_clock = next(self.__clock)
                return f"{next_clock}"
            case "arrows":
                next_arrows = next(self.__arrows)
                return f"{next_arrows}"
            case "triangles":
                next_triangles = next(self.__triangles)
                return f"{next_triangles}"
            case "cross":
                next_cross = next(self.__cross)
                return f"{next_cross}"
            case "moon":
                next_moon = next(self.__moon)
                return f"{next_moon}"
            case _:
                raise ValueError(
                    f"Invalid progress_type: {progress_type}. Choose from ['bar', 'pie', 'squares', 'bounce', 'clock', 'arrows', 'triangles', 'cross', 'moon']."
                )

    def __progress_contents(
        self, time_estimate: TimeEstimate, metrics: List[Tuple[str, float]] | None
    ) -> str:
        """
        Returns the progress bar string with metrics.

        Args:
            metrics (List[Tuple[str, float]]): List of tuples containing metric name and value.


        Returns:
            str: Formatted string of progress bar.
        """
        formatted_metrics = None

        if metrics is not None:
            formatted_metrics = " - ".join(
                [f"{name}: {value:.4f}" for name, value in metrics]
            )

        # Calculate the number of filled and empty slots in the progress bar
        filled_length = int(self.__bar_width * self.__current_value // self.__total)
        empty_length = self.__bar_width - filled_length
        # Bar styles
        bar_style = self.__bar_style()
        bar = self.__progress_symbol(
            progress_type=self.__progress_type,
            filled_length=filled_length,
            empty_length=empty_length,
            bar_style=bar_style,
        )
        percent = int((self.__current_value / self.__total) * 100)
        self.__percent = percent

        elapsed = f"{time_estimate.elapsed}s"
        milliseconds = f"{time_estimate.milliseconds}ms/step"

        return self.__selecting_verbose(
            bar=bar,
            elapsed=elapsed,
            milliseconds=milliseconds,
            formatted_metrics=formatted_metrics,
            percent=percent,
        )

    def __render(
        self, time_estimate: TimeEstimate, metrics: List[Tuple[str, float]] | None
    ):
        """
        Renders the progress bar with metrics.

        Args:
            metrics (List[Tuple[str, float]]): List of tuples containing metric name and value.
        """
        content = self.__progress_contents(time_estimate, metrics)

        if (
            self.__verbose is not None
            and self.__verbose != "only_epochs"
            and self.__verbose != 0
        ):
            sys.stdout.write(content)
            sys.stdout.flush()

    def update(
        self, current_value: int, metrics: List[Tuple[str, float]] | None = None
    ):
        """
        Updates the progress bar with new metrics.

        Args:
            metrics (List[Tuple[str, float]]): List of tuples containing metric name and value.
        """
        self.__current_value = current_value

        time_estimate = self.__time_estimate(current_value, self.__start_time)
        self.__render(time_estimate, metrics)

        if self.__current_value == self.__total:
            self.__last_time_estimate = time_estimate
            self.__start_time = time.time()
            self.__reset_cycles

    def __reset_progress(self):
        """
        Resets the progress bar.
        """
        self.__start_time = time.time()
        self.__last_update_start_time = time.time()
        self.__reset_cycles

    def last_update(self, current_value, metrics: List[Tuple[str, float]]):
        """
        Finishes the progress bar and resets it.
        """
        time_estimate = self.__time_estimate(
            current_value, self.__last_update_start_time
        )
        time_estimate = TimeEstimate(
            elapsed=self.__last_time_estimate.elapsed + time_estimate.elapsed,
            milliseconds=self.__last_time_estimate.milliseconds
            + time_estimate.milliseconds,
        )

        self.__render(time_estimate, metrics)
        sys.stdout.flush()
        self.__reset_progress()

        if self.__pre_epoch == self.__epochs:
            print("\n")


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
        self._model: nn.Module | None = None
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
        self._model = nn.Sequential(*self.layers)

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
        self.optimizer = self.optimizer_obj(self._model.parameters())
        self._model = self._model.to(self._device)

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
        
        # validation data size
        val_data_size = len(val_data.dataset)
        
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

        if self._model is None:
            raise ValueError("Model is not initialized.")

        # Indicate the model to train
        self._model.train()

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
            predict = self._model(feature).float()

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

        if self._model is None:
            raise ValueError("Model is not initialized.")

        # Indicate the model to evaluate
        self._model.eval()

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
                predict = self._model(feature).float()

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
 