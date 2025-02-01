# Praise The LORD of Host.

"""
======== ExtTorch Callbacks implementation ========
"""

# Import libraries
from exttorch._callbacks import Callback
from typing import Dict

class EarlyStopping(Callback):
    def on_batch_end(self, logs):
        print(logs)