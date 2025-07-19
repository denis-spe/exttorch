# Bless be the LORD of host.

# Import libraries.
from typing import Dict, Any, List, Iterator, Tuple, Literal
import torch.nn as __nn__
from types import GeneratorType
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torch import TensorType
from numpy.typing import ArrayLike
from torch import nn as __nn__
from sklearn.utils import Bunch

Logs = Dict[str, float] | None
Weight = Dict[str, Any]
Layer = __nn__.Module
Layers = List[Layer] | None
Loss = __nn__.CrossEntropyLoss | __nn__.MSELoss | __nn__.BCELoss | __nn__.NLLLoss
Dataset_DataLoader_TensorDataset_ArrayLike_Subset_Iterator_TensorType_Bunch = (
    Dataset[Any]
    | DataLoader[Any]
    | TensorDataset
    | ArrayLike
    | Subset[Any]
    | Iterator
    | TensorType
    | Bunch
)
List_Tuple_DataLoader_Dataset_TensorDataset = (
    None
    | List[ArrayLike | Bunch]
    | Tuple[ArrayLike | Bunch, ArrayLike]
    | DataLoader
    | Dataset
    | TensorDataset
)
FillStyleType = Literal["━", "◉", "◆", "●", "█", "▮", "=", "#", "▶", "■", "➤"]
EmptyStyleType = Literal["━", "◎", "◇", "○", "░", "▯", "-", "▒", ".", "▷", "□"]
ProgressType = Literal[
    "bar", "pie", "squares", "cross", "arrows", "clock", "bounce", "moon", "triangles"
]
VerboseType = Literal[
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
    "hide-time-estimation",
    "percentage",
    "only_percentage",
    "only_epochs",
    "only_batch_size",
    "only_metrics",
    "only_train_metrics",
    "only_val_metrics",
    "only_progress_bar",
    "only_time_estimation",
]
