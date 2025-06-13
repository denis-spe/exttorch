# Bless be the LORD of host.

# Import libraries.
from typing import Dict, Any, List, NewType
from torch.nn import Module
from types import GeneratorType
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch import TensorType
from numpy.typing import ArrayLike

Logs = Dict[str, float] | None
Weight = Dict[str, Any]
Layer = Module
Layers = List[Layer] | None
Data = Dataset[Any] | DataLoader[Any] | TensorDataset | ArrayLike | Subset[Any] | GeneratorType | TensorType

