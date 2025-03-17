# Bless be the Name of the LORD

# import libraries
import torch
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def __str__(self):
        ...
    
    @abstractmethod
    def __call__(self):
        ...

class NLLLoss(Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        self.__weight = weight
        self.__size_average = size_average
        self.__ignore_index = ignore_index
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "NLLLoss"
    
    def __call__(self):
        return torch.nn.NLLLoss(
            weight=self.__weight,
            size_average=self.__size_average,
            ignore_index=self.__ignore_index,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class CrossEntropyLoss(Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        self.__weight = weight
        self.__size_average = size_average
        self.__ignore_index = ignore_index
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "CrossEntropyLoss"
    
    def __call__(self):
        return torch.nn.CrossEntropyLoss(
            weight=self.__weight,
            size_average=self.__size_average,
            ignore_index=self.__ignore_index,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class MSELoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        self.__size_average = size_average
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "MSELoss"
    
    def __call__(self):
        return torch.nn.MSELoss(
            size_average=self.__size_average,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class L1Loss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        self.__size_average = size_average
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "L1Loss"
    
    def __call__(self):
        return torch.nn.L1Loss(
            size_average=self.__size_average,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class BCELoss(Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        self.__weight = weight
        self.__size_average = size_average
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "BCELoss"
    
    def __call__(self):
        return torch.nn.BCELoss(
            weight=self.__weight,
            size_average=self.__size_average,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class BCEWithLogitsLoss(Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        self.__weight = weight
        self.__size_average = size_average
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "BCEWithLogitsLoss"
    
    def __call__(self):
        return torch.nn.BCEWithLogitsLoss(
            weight=self.__weight,
            size_average=self.__size_average,
            reduce=self.__reduce,
            reduction=self.__reduction
            )
        
class MarginRankingLoss(Loss):
    def __init__(self, margin=0, size_average=None, reduce=None, reduction='mean'):
        self.__margin = margin
        self.__size_average = size_average
        self.__reduce = reduce
        self.__reduction = reduction
        
    def __str__(self):
        return "MarginRankingLoss"
    
    def __call__(self):
        return torch.nn.MarginRankingLoss(
            margin=self.__margin,
            size_average=self.__size_average,
            reduce=self.__reduce,
            reduction=self.__reduction
            )