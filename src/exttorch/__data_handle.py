# Praise Ye The Lord

import types
from typing import Tuple, Optional, Any, Iterator

import numpy as np
import pandas as pd
# Import the libraries
import torch
from numpy.typing import ArrayLike as __ArrayLike__
from sklearn.utils import Bunch as __Bunch__
from torch.utils import data as __data__


class DataHandler:
    def __init__(
        self,
        x: __data__.Dataset[Any] | __data__.DataLoader[Any] | __data__.TensorDataset | __ArrayLike__ | __data__.Subset[Any] | Iterator | torch.TensorType | __Bunch__,
        y:  __ArrayLike__ | __Bunch__ | None = None,
        batch_size: int = 1,
        val_batch_size: int = 1,
        shuffle: bool = False,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> None:

        # Import the libraries
        import torch

        if isinstance(x, pd.DataFrame):
            self.__x = x
        else:
            self.__x = x
        if isinstance(y, pd.Series):
            self.__y = y.reshape(-1)
        else:
            self.__y = y.reshape(-1) if y is not None else None
        self.__batch_size = batch_size
        self.__val_batch_size = val_batch_size
        self.__shuffle = shuffle
        self.__kwargs = kwargs
        self.__generator = None
        if random_seed is not None:
            self.__generator = torch.Generator()
            self.__generator.manual_seed(random_seed)

    def __split_data(self, data: Any, val_size: float):
        """
        Split the data into train and validation data.
        """
        from torch.utils.data import random_split

        # Check if the data is a DataLoader object
        dataset = data.dataset if isinstance(data, __data__.DataLoader) else data

        # Split the data into train and validation.
        data_split = random_split(
            dataset, lengths=[1 - val_size, val_size], generator=self.__generator
        )

        return (sample for sample in data_split)

    def __call__(
        self, val_size: Optional[float] = None
    ) -> __data__.DataLoader | Tuple[__data__.DataLoader, __data__.DataLoader] | __data__.Dataset | Iterator:
        from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

        if isinstance(self.__x, np.ndarray) and isinstance(self.__y, np.ndarray):
            # Change x and y to torch tensor
            x_tensor = torch.from_numpy(self.__x)
            y_tensor = torch.from_numpy(self.__y)

            # Create a __TensorDataset object
            __Dataset_obj = TensorDataset(x_tensor, y_tensor)

            if val_size is not None:
                train_data, val_data = self.__split_data(__Dataset_obj, val_size)
                return (
                    DataLoader(
                        train_data,
                        batch_size=self.__batch_size,
                        shuffle=self.__shuffle,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                    DataLoader(
                        val_data,
                        batch_size=self.__val_batch_size,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                )

            return DataLoader(
                __Dataset_obj,
                batch_size=self.__batch_size,
                generator=self.__generator,
                **self.__kwargs,
            )

        elif isinstance(self.__x, Subset):
            return self.__x.dataset

        elif isinstance(self.__x, Dataset) or isinstance(self.__x, TensorDataset):
            if val_size is not None:
                train_data, val_data = self.__split_data(self.__x, val_size)
                return (
                    DataLoader(
                        train_data,
                        batch_size=self.__batch_size,
                        shuffle=self.__shuffle,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                    DataLoader(
                        val_data,
                        batch_size=self.__val_batch_size,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                )

            return DataLoader(
                self.__x,
                batch_size=self.__batch_size,
                generator=self.__generator,
                **self.__kwargs,
            )

        elif isinstance(self.__x, DataLoader):
            if val_size is not None:
                train_data, val_data = self.__split_data(self.__x, val_size)

                return (
                    DataLoader(
                        train_data,
                        batch_size=self.__batch_size,
                        shuffle=self.__shuffle,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                    DataLoader(
                        val_data,
                        batch_size=self.__val_batch_size,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                )

            return self.__x

        elif isinstance(self.__x, types.GeneratorType):
            x, y = next(self.__x)

            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)

            __Dataset_obj = None
    
            if isinstance(x, torch.Tensor):
                __Dataset_obj = TensorDataset(x, y)
            else:
                raise TypeError(
                    "The generator must yield a tuple of (x, y) where x is a torch.Tensor or np.ndarray and y is a torch.Tensor or np.ndarray."
                )

            if val_size is not None:
                train_data, val_data = self.__split_data(__Dataset_obj, val_size)
                return (
                    DataLoader(
                        train_data,
                        batch_size=self.__batch_size,
                        shuffle=self.__shuffle,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                    DataLoader(
                        val_data,
                        batch_size=self.__val_batch_size,
                        generator=self.__generator,
                        **self.__kwargs,
                    ),
                )

            return DataLoader(
                __Dataset_obj,
                batch_size=self.__batch_size,
                generator=self.__generator,
                **self.__kwargs,
            )
        else:
            raise ValueError(
                f"Invalid data of type {type(self.__x)} for x, expected type of "
                + "`np.ndarray | DataLoader | Dataset | TensorDataset` for x "
                + "and np.ndarray for y"
            )

    def data_preprocessing(self, nprocs: int, val_size: Optional[float] = None):
        dataloader = self.__call__(val_size=val_size)
        
        return dataloader
