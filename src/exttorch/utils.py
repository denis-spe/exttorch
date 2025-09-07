# Holy, holy, holy is the LORD of host, the whole earth is full of his glory.

# Importing necessary modules

import sys
import time
from typing import List, Tuple
from dataclasses import dataclass
import itertools
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