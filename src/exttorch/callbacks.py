# Praise The LORD of Host.

"""
======== ExtTorch Callbacks implementation ========
"""

# Import libraries
from typing import Dict as __Dict__
import numpy as __np__
from typing import Literal as __Literal__
from enum import Enum as __Enum__

class Callback:
    def __init__(self):
        self.model = None
        
    def on_train_begin(self) -> None:
        ...
    
    def on_train_end(self, logs: __Dict__) -> None:
        ...
        
    def on_epoch_begin(self, epoch: int) -> None:
        ...
        
    def on_epoch_end(self, epoch: int, logs: __Dict__) -> None:
        ...
        
    def on_validation_begin(self) -> None:
        ...
        
    def on_validation_end(self, logs: __Dict__) -> None:
        ...
        
    def on_batch_begin(self) -> None:
        ...
        
    def on_batch_end(self, logs: __Dict__) -> None:
        ...
        
class __MetricInitial__(__Enum__):
    DECREASING = __np__.inf
    INCREASING = 0

class EarlyStopping(Callback):
    def __init__(
        self, 
        patience: int=0, 
        monitor: str = "loss", 
        mode: __Literal__["auto", "min", "max"] = "auto"
        ):
        super().__init__()
        self.__patience = patience
        self.__monitor = monitor
        self.__mode_str = mode
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = self.__metric_state.value
    
    @property
    def __metric_state(self) -> __MetricInitial__:
        monitor = self.__monitor.removeprefix("val_")
        
        match monitor:
            # Increase in metric
            case monitor if monitor in ["acc", "Accuracy"]:
                return __MetricInitial__.INCREASING
            # Decrease in metric
            case monitor if monitor in ["loss"]:
                return __MetricInitial__.DECREASING
                
        
    def __mode(self, mode: str, current: float) -> __np__.ndarray[bool]:
        if mode == "min":
            return __np__.less(current, self.best)
        elif mode == "max":
            return __np__.greater(current, self.best)
        else:
            raise ValueError(f"Invalid mode {mode}")
        
    def __check_state(self, current: float) -> __np__.ndarray[bool]:
        if self.__mode_str == "auto":
            if self.__metric_state == __MetricInitial__.INCREASING:
                return self.__mode("max", current=current)
            if self.__metric_state == __MetricInitial__.DECREASING:
                    return self.__mode("min", current=current)
        else:
            return self.__mode(self.__mode_str, current=current)

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.__monitor)
        
        if self.__check_state(current):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.__patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\nRestoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping\n")