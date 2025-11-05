# Praise Ye The Lord

# Import libraries
import types
from collections.abc import Sized
from typing import Tuple, Optional, Any, Iterator, List, Dict, Union, TypeAlias

import numpy as np
import pandas as pd
import torch
from sklearn.utils import Bunch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, random_split

Xdata: TypeAlias = Union[DataLoader[Any], Dataset[Any], TensorDataset, np.ndarray, Bunch, pd.DataFrame, torch.Tensor, Sized]
Ydata = Union[np.ndarray, Bunch, pd.Series, torch.Tensor, None]
ValidationData = Union[Iterator[np.ndarray], Iterator[torch.Tensor], Iterator[Bunch], DataLoader[Any], None]

class DataHandler:
    def __init__(
        self,
        x: Xdata, 
        y: Ydata,
        dataloader_kwargs: Dict[str, Any],
        val_dataloader_kwargs: Dict[str, Any],
        validation_data: ValidationData = None,
        validation_split: float | None = None,
        generator: Optional[torch.Generator] = None,
        ) -> None:
        self.__x = x
        self.__y = y
        self.__validation_data = validation_data
        self.__validation_split = validation_split
        self.__dataloader_kwargs = dataloader_kwargs
        self.__val_dataloader_kwargs = val_dataloader_kwargs
        self.__generator = generator
        if self.__generator is not None:
            self.__dataloader_kwargs['generator'] = self.__generator
            self.__val_dataloader_kwargs['generator'] = self.__generator

    def __len__(self) -> int:
        data = self.__handle_x_y_data()
        dataset = data.dataset
        data_len = len(dataset)
        return data_len

    def __handle_x_y_data(self) -> DataLoader:
        if isinstance(self.__x, DataLoader) and self.__y is not None:
            raise ValueError("y must be none, if used dataloader on x")
        else:
            # ... DataLoader
            if isinstance(self.__x, DataLoader):
                return self.__x
            
            # ... Dataset
            if isinstance(self.__x, Dataset):
                return DataLoader(self.__x, **self.__dataloader_kwargs)
            
            # ... TensorDataset
            elif isinstance(self.__x, TensorDataset):
                return DataLoader(self.__x, **self.__dataloader_kwargs)
            
            # ... Ndarray
            elif isinstance(self.__x, np.ndarray) and isinstance(self.__y, np.ndarray):
                # Numpy to tensor
                x = torch.from_numpy(self.__x)
                y = torch.from_numpy(self.__y)
                
                # Create tensor dataset
                tensor_data = TensorDataset(x, y)
                
                return DataLoader(tensor_data, **self.__dataloader_kwargs)
            
            # ... Bunch
            elif isinstance(self.__x, Bunch) and isinstance(self.__y, Bunch):
                # Numpy to tensor
                x = torch.from_numpy(self.__x)
                y = torch.from_numpy(self.__y)
                
                # Create tensor dataset
                tensor_data = TensorDataset(x, y)
                
                return DataLoader(tensor_data, **self.__dataloader_kwargs)
            
            # ... DataFrame with Series
            elif isinstance(self.__x, pd.DataFrame) and isinstance(self.__y, pd.Series):
                # Numpy to tensor
                x = torch.from_numpy(self.__x.values)
                y = torch.from_numpy(self.__y.values)
                
                # Create tensor dataset
                tensor_data = TensorDataset(x, y)
                
                return DataLoader(tensor_data, **self.__dataloader_kwargs)
            
            # ... Tensor
            elif isinstance(self.__x, torch.Tensor) and isinstance(self.__y, torch.Tensor):
                
                # Create tensor dataset
                tensor_data = TensorDataset(self.__x, self.__y)
                
                return DataLoader(tensor_data, **self.__dataloader_kwargs)
            
            # ... Generator
            elif isinstance(self.__x, types.GeneratorType) and self.__y is None:
                x, y = next(self.__x)
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x)
                if not isinstance(y, torch.Tensor):
                    y = torch.from_numpy(y)

                gen_dataset = TensorDataset(x, y)
                return DataLoader(gen_dataset, **self.__dataloader_kwargs)
            
            else:
                raise ValueError("Expected x to be DataLoader, Dataset, TensorDataset, np.ndarray, "
                                 f"Bunch, pd.DataFrame or torch.Tensor but got `{self.__x}`"
                                 " and y to be np.ndarray, Bunch, pd.Series or torch.Tensor or None"
                                 )

    def __handle_validation_data(self) -> Optional[DataLoader]:
        if isinstance(self.__validation_data, (Tuple, List)):
            x, y = self.__validation_data
            
            # ... Tensor
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                # Create tensor dataset
                tensor_data = TensorDataset(x, y)
                
                return DataLoader(tensor_data, **self.__val_dataloader_kwargs)
            
            # ... Ndarray
            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                # Numpy to tensor
                x_tensor = torch.from_numpy(x)
                y_tensor = torch.from_numpy(y)
                
                # Create tensor dataset
                tensor_data = TensorDataset(x_tensor, y_tensor)
                
                return DataLoader(tensor_data, **self.__val_dataloader_kwargs)
            
            # ... Bunch
            elif isinstance(x, Bunch) and isinstance(y, Bunch):
                x_tensor = torch.from_numpy(x)
                y_tensor = torch.from_numpy(y)
                
                # Create tensor dataset
                tensor_data = TensorDataset(x_tensor, y_tensor)
                
                return DataLoader(tensor_data, **self.__val_dataloader_kwargs)
            else:
                raise ValueError("Expected validation_data to be Tuple or List of (np.ndarray, np.ndarray), "
                                 "(Bunch, Bunch) or (torch.Tensor, torch.Tensor)")
        elif isinstance(self.__validation_data, DataLoader):
            return self.__validation_data
        elif self.__validation_data is None:
            return None
        else:
            raise ValueError("Expected validation_data to be Tuple, List or DataLoader or None")
    
    def __handle_x_split(self) -> List[Subset[Any]] | None:
        if (self.__validation_split is not None and
            (self.__validation_split > 0 and self.__validation_split < 1)
            ):
            data = self.__handle_x_y_data()
            dataset = data.dataset
            data_len = len(self)

            train_size = int(data_len * (1 - self.__validation_split))
            val_size = data_len - train_size
            return random_split(dataset, [train_size, val_size], generator=self.__generator)
        
        return None
    
    def get_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        if self.__validation_data is not None:
            train_data = self.__handle_x_y_data()
            val_data = self.__handle_validation_data()
            return train_data, val_data
        elif self.__validation_split is not None:
            split_data = self.__handle_x_split()
            if split_data is not None:
                train_subset, val_subset = split_data
                train_data = DataLoader(train_subset, **self.__dataloader_kwargs)
                val_data = DataLoader(val_subset, **self.__val_dataloader_kwargs)
                return train_data, val_data
            else:
                raise ValueError("validation_split must be between 0 and 1")
        else:
            train_data = self.__handle_x_y_data()
            return train_data, None