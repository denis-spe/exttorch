# Love the LORD your GOD with all your mind and with all your  heart and with all your soul
# And Love your neighbor as your self

from src.exttorch.__data_handle import DataHandler, ValidationData, Xdata, Ydata
from src.exttorch.__metrics_handles import MetricStorage
from src.exttorch.__types import (
    VerboseType,
    FillStyleType,
    EmptyStyleType,
    ProgressType,
    Weight,
)
from typing import Dict, List
from src.exttorch.history import History

class ModelInf:
    def __init__(self) -> None:
        self.stop_training: bool = False
        
    def fit(
        self,
        x: Xdata,
        y: Ydata = None,
        *,
        epochs: int = 1,
        random_seed: int | None = None,
        shuffle: bool = False,
        batch_size: int | None = 1,
        val_batch_size: int | None = 1,
        validation_split: float | None = None,
        validation_data=None,
        callbacks=None,
        progress_bar_width: int = 40,
        progress_fill_style: FillStyleType = "━",
        progress_empty_style: EmptyStyleType = "━",
        progress_fill_color: str = "\033[92m",
        progress_empty_color: str = "\033[90m",
        progress_percentage_colors=None,
        progress_progress_type: ProgressType = "bar",
        verbose: VerboseType = "full",
        val_dataloader_kwargs=None,
        **dataloader_kwargs,
    ) -> History: ...

    def evaluate(
        self,
        x,
        y=None,
        batch_size: int | None = 1,
        progress_bar_width: int = 40,
        progress_fill_style: FillStyleType = "━",
        progress_empty_style: EmptyStyleType = "━",
        progress_fill_color: str = "\033[92m",
        progress_empty_color: str = "\033[90m",
        progress_percentage_colors=None,
        progress_progress_type: ProgressType = "bar",
        verbose: VerboseType = "full",
        **dataloader_kwargs,
    ) -> Dict | List: ...

    def train_stage(
        self,
        train_data,
    ) -> dict: ...
    
    def model_state_dict(self) -> Weight: ...
    
    def load_model_state_dict(self, weight: Weight): ...