# Praise Ye The Lord

# Import the libraries
import torch
import numpy as np
import pandas as pd
from typing import Any, Optional


class SinglePredictionsFormat:
    def __init__(self, prediction):
        self.__prediction = prediction
        self.__size = (prediction.size()
                        if isinstance(prediction, torch.Tensor)
                        else prediction.shape)

    def __single_format(self, prediction):
        if self.__size[1] > 1:
            # That's a category prediction
            return (torch.argmax(prediction)
                    if isinstance(prediction, torch.Tensor)
                    else np.argmax(prediction))
        # else it's a continuous prediction
        return prediction

    def format_prediction(self) -> Any:
        if self.__size[0] > 1:
            # It's a batched prediction
            return self.__batched_prediction()
        # It's a single prediction
        return self.__single_format(self.__prediction)

    def __batched_prediction(self) -> torch.Tensor:
        return torch.tensor(list(map(
            lambda tensor: self.__single_format(tensor),
            self.__prediction
        ))).view(-1, 1)


class DataHandler:
    def __init__(self,
                x: Any,
                y: Any = None,
                batch_size: Optional[int] = 1,
                shuffle: bool = False,
                generator: Optional[int] = None,
                 **kwargs) -> None:
        if isinstance(x, pd.DataFrame):
            self.__x = x.to_numpy()
        else:
            self.__x = x
        if isinstance(y, pd.Series) and y is not None:
            self.__y = y.to_numpy().reshape(-1)
        else:
            self.__y = y.reshape(-1) if y is not None else None
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__kwargs = kwargs
        self.__generator = generator

    def __split_data(self,
                    data: Any,
                    val_size: float):
        """
        Split the data into train and validation data.
        """
        from torch.utils.data import random_split

        # Split the data into train and validation.
        data_split = random_split(
            data,
            lengths=[1 - val_size, val_size],
            generator=self.__generator)

        return (sample for sample in data_split)

    def __call__(self, val_size: Optional[float] = None) -> Any:
        from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

        if (isinstance(self.__x, np.ndarray) and
                isinstance(self.__y, np.ndarray)):
            # Change x and y to torch tensor
            x_tensor = torch.from_numpy(self.__x)
            y_tensor = torch.from_numpy(self.__y)

            # Create a __TensorDataset object
            __Dataset_obj = TensorDataset(x_tensor, y_tensor)

            if val_size is not None:
                train_data, val_data = self.__split_data(__Dataset_obj, val_size)
                return (DataLoader(train_data,
                                    shuffle=self.__shuffle,
                                    generator=self.__generator,
                                   **self.__kwargs),
                        DataLoader(val_data,
                                    generator=self.__generator,
                                    **self.__kwargs))

            return DataLoader(__Dataset_obj,
                            generator=self.__generator,
                            **self.__kwargs)

        elif isinstance(self.__x, Subset):
            return self.__x.dataset

        elif (isinstance(self.__x, Dataset) or
            isinstance(self.__x, TensorDataset)):
            if val_size is not None:
                train_data, val_data = self.__split_data(self.__x, val_size)
                return (DataLoader(train_data,
                                    shuffle=self.__shuffle,
                                    generator=self.__generator,
                                   **self.__kwargs),
                        DataLoader(val_data,
                                    generator=self.__generator,
                                    **self.__kwargs))

            return DataLoader(self.__x,
                            generator=self.__generator,
                                **self.__kwargs)

        elif isinstance(self.__x, DataLoader):
            if val_size is not None:
                train_data, val_data = self.__split_data(self.__x, val_size)

                return (DataLoader(train_data,
                                    shuffle=self.__shuffle,
                                    generator=self.__generator,
                                   **self.__kwargs),
                        DataLoader(val_data,
                                    generator=self.__generator,
                                   **self.__kwargs))

            return self.__x
        else:
            raise ValueError("Invalid data, expected type of " +
                            "`np.ndarray | DataLoader | Dataset | TensorDataset` for x " +
                            "and np.ndarray for y")
