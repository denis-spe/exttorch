# Praise Ye The Lord

# Import the libraries
import torch
from typing import Any, Optional
import numpy as np
import pandas as pd
from torch.utils.data import (
    Dataset, Subset, DataLoader, random_split, TensorDataset)


class SinglePredictionsFormat:
    def __init__(self, prediction):
        self.__prediction = prediction
        self.__size = (prediction.size()
        if type(prediction) == torch.Tensor else prediction.shape)

    def __single_format(self, pred):
        if self.__size[1] > 1:
            # That's a category prediction
            return (torch.argmax(pred)
                if type(pred) == torch.Tensor else np.argmax(pred))
        # else its a continuous prediction
        return pred

    def format_prediction(self) -> Any:
        if self.__size[0] > 1:
            # It's a batched prediction
            return self.__batched_pred()
        # It's a single prediction
        return self.__single_format(self.__prediction)


    def __batched_pred(self) -> torch.Tensor:
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
            generator: Optional[torch.Generator] = None,
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

        # Split the data into train and validation.
        data_split = random_split(
            data,
            lengths=[1 - val_size, val_size],
            generator=self.__generator)

        return (sample for sample in data_split)


    def __call__(self, val_size: Optional[float] = None) -> Any:
        if (isinstance(self.__x, np.ndarray) and
            isinstance(self.__y, np.ndarray)):
            # Change x and y to torch tensor
            x_tensor = torch.from_numpy(self.__x)
            y_tensor = torch.from_numpy(self.__y)

            # Create a TensorDataset object
            dataset_obj = TensorDataset(x_tensor, y_tensor)

            if val_size is not None:
                train_data, val_data = self.__split_data(dataset_obj, val_size)
                return (DataLoader(train_data,
                                    shuffle=self.__shuffle,
                                    generator=self.__generator,
                                   **self.__kwargs),
                        DataLoader(val_data,
                                    generator=self.__generator,
                                   **self.__kwargs))

            return DataLoader(dataset_obj,
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
            raise ValueError("Invalid data, expected type of "+
            "`np.ndarray | DataLoader | Dataset|TensorDataset` for x " +
            "and np.ndarray for y")