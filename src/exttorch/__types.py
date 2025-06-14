# Bless be the LORD of host.

# Import libraries.
from typing import Dict, Any, List, NewType
import torch.nn as __nn__
from types import GeneratorType
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch import TensorType
from numpy.typing import ArrayLike

Logs = Dict[str, float] | None
Weight = Dict[str, Any]
Layer = __nn__.Module
Layers = List[Layer] | None
Loss = __nn__.CrossEntropyLoss | __nn__.MSELoss | __nn__.BCELoss | __nn__.NLLLoss

