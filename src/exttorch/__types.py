# Bless be the LORD of host.

# Import libraries.
from typing import Dict, Any, List
from torch.nn import Module
from exttorch.callbacks import Callback

Logs = Dict[str, float] | None
Weight = Dict[str, Any]
Layer = Module
Layers = List[Layer]
Callbacks = List[Callback]