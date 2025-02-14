# Praise The LORD of Host.

"""
======== ExtTorch Callbacks implementation ========
"""

# Import libraries
from typing import Dict

from typing import Dict

class Callback:
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
    def on_batch_end(self, logs):
        pass
        # print(logs)