# Praise The LORD of Host.

"""
======== ExtTorch Callbacks implementation ========
"""

# Import libraries
from typing import Dict
import numpy as np

class Callback:
    def __init__(self):
        self.model = None
        
    def on_train_begin(self) -> None:
        ...
    
    def on_train_end(self, logs: Dict) -> None:
        ...
        
    def on_epoch_begin(self, epoch: int) -> None:
        ...
        
    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        ...
        
    def on_validation_begin(self) -> None:
        ...
        
    def on_validation_end(self, logs: Dict) -> None:
        ...
        
    def on_batch_begin(self) -> None:
        ...
        
    def on_batch_end(self, logs: Dict) -> None:
        ...

class EarlyStopping(Callback):
    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")