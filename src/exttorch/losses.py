""" Bless be the Name of the LORD """

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
        
def __change_str_to_loss(loss: str):
    match loss:
        case "MSELoss" | "mse" | "mean_squared_error" | "MSE":
            return MSELoss()
        case "L1Loss" | "l1" | "mean_absolute_error" | "MAE":
            return L1Loss()
        case "NLLLoss" | "nll" | "negative_log_likelihood" | "nll_loss":
            return NLLLoss()
        case "CrossEntropyLoss" | "cross_entropy" | "crossentropy":
            return CrossEntropyLoss()
        case "BCELoss" | "bce" | "binary_cross_entropy":
            return BCELoss()
        case "BCEWithLogitsLoss" | "bce_with_logits" | "binary_cross_entropy_with_logits":
            return BCEWithLogitsLoss()
        case "MarginRankingLoss" | "margin_ranking":
            return MarginRankingLoss()
        case _:
            raise ValueError("Invalid loss function")

class NLLLoss(Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        """
        Negative Log Likelihood Loss for classification problems.
        Args:
            weight: A tensor of size C, where C is the number of classes. This is used to weigh the loss for each class.
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        This criterion combines LogSoftmax and NLLLoss in one single class.
        It is useful when training a classification problem with C classes.
        If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes.
        This is particularly useful when you have an unbalanced training set.
        The input given through a forward call is expected to contain log-probabilities of each class.
        Args:
            weight: A tensor of size C, where C is the number of classes. This is used to weigh the loss for each class.
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        This criterion creates a criterion that measures the mean squared error between each element in the input x and target
        Args:
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target
        Args:
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        Creates a criterion that measures the Binary Cross Entropy between the target and the output
        Args:
            weight: A tensor of size C, where C is the number of classes. This is used to weigh the loss for each class.
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        This loss combines a Sigmoid layer and the BCELoss in one single class.
        This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
        Args:
            weight: A tensor of size C, where C is the number of classes. This is used to weigh the loss for each class.
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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
        """
        Creates a criterion that measures the loss given an input tensors x1, x2 and a label tensor y with values 1 or -1.
        This is used for measuring whether two inputs are similar or dissimilar, using the margin parameter.
        If y == 1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y == -1.
        The loss function for each sample in the mini-batch is:
        loss(x, y) = max(0, -y * (x1 - x2) + margin)
        Args:
            margin: Has a default value of 0
            size_average: Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there are multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
            reduce: Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
        """
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