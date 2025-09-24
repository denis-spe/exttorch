# Praise The LORD of Host.

"""
======== ExtTorch Callbacks implementation ========
"""

from dataclasses import dataclass
from typing import Literal as __Literal__
import numpy as __np__

# Import libraries
from src.exttorch import __types as __types__
# from src.exttorch.__model import Model as __Model


class Callback:
    def __init__(self):
        self.model = None
        self.monitor: str = "loss"

    def on_train_begin(self) -> None: ...

    def on_train_end(self, logs: __types__.Logs) -> None: ...

    def on_epoch_begin(self, epoch: int) -> None: ...

    def on_epoch_end(self, epoch: int, logs: __types__.Logs) -> None: ...

    def on_validation_begin(self) -> None: ...

    def on_validation_end(self, logs: __types__.Logs) -> None: ...

    def on_batch_begin(self) -> None: ...

    def on_batch_end(self, logs: __types__.Logs) -> None: ...

    @property
    def metric_state(self) -> float:
        monitor = self.monitor.removeprefix("val_")

        match monitor:
            # Increase in metric
            case monitor if monitor in ["acc", "Accuracy"]:
                return MetricInitial.INCREASING
            # Decrease in metric
            case monitor if monitor in ["loss"]:
                return MetricInitial.DECREASING
            case _:
                raise ValueError(f"Invalid monitor name `{monitor}`")


@dataclass
class MetricInitial:
    DECREASING: float = __np__.inf
    INCREASING: float = 0.0


class EarlyStopping(Callback):
    def __init__(
        self,
        patience: int = 0,
        monitor: str = "loss",
        mode: __Literal__["auto", "min", "max"] = "auto",
    ):
        super().__init__()
        self.best = None
        self.stopped_epoch = None
        self.wait = None
        self.__patience = patience
        self.__monitor = monitor
        self.__mode_str = mode
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights: __types__.Weight = dict()

    def on_train_begin(self, logs: __types__.Logs | None = None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best: float = self.metric_state

        print(
            f"\nEarlyStopping will monitor `{self.__monitor}` with mode `{self.__mode_str}`"
        )

    def __mode(self, mode: str, current: float) -> bool:
        if mode == "min":
            return __np__.less(current, self.best)
        elif mode == "max":
            return __np__.greater(current, self.best)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def __check_state(self, current: float) -> bool:
        if self.__mode_str == "auto":
            if self.metric_state == MetricInitial.INCREASING:
                return self.__mode("max", current=current)
            if self.metric_state == MetricInitial.DECREASING:
                return self.__mode("min", current=current)
            else:
                raise ValueError("Invalid value")
        else:
            return self.__mode(self.__mode_str, current=current)

    def on_epoch_end(self, epoch: int, logs: __types__.Logs | None = None):
        assert logs is not None, "Logs were not provided."
        assert self.model is not None, "Model must be set before calling on_epoch_end."

        current = logs.get(self.__monitor)

        if current is not None and self.__check_state(current):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.model_state_dict()
        else:
            self.wait += 1
            if self.wait >= self.__patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\nRestoring model weights from the end of the best epoch.")
                self.model.load_model_state_dict(self.best_weights)

    def on_train_end(self, logs: __types__.Logs = None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping\n")


class SaveOnCheckpoint(Callback):

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: __Literal__["auto", "min", "max"] = "auto",
        save_freq: str = "epoch",
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq

    def on_epoch_end(self, epoch: int, logs: __types__.Logs | None = None):
        assert logs is not None, "Logs were not provided."
        assert self.model is not None, "Model must be set before calling on_epoch_end."

        current = logs.get(self.monitor)

        if current is None:
            return
