# Praise Ye The Lord

# Import libraries
from abc import ABCMeta as __abc__, abstractmethod as __abs__
import numpy as np
from sklearn import metrics
from typing import Literal
from numpy.typing import ArrayLike
import torch

def __mean__(value: np.ndarray, rounded_by = 4):
    return np.array([np.mean(value).round(decimals=rounded_by)])


def __return_value__(value: float | np.ndarray) -> np.ndarray:
    """
    Returns the value as a numpy array
    """
    if isinstance(value, np.floating):
        return round(value, 4)
    elif isinstance(value, float):
        return np.array([value]).round(decimals=4)
    else:
        raise TypeError("Value must be a numpy array or float")
    

def __restrict_if_nunique__(
    y: np.ndarray,
    prediction: np.ndarray,
    num_classes: int = 2,
):
    y_unique = np.unique(y)
    pred_unique = np.unique(prediction)
    
    if len(prediction.shape) == 2 and prediction.shape[1] > 1 and prediction.shape[1] != len(y_unique):
        return True
    
    if len(y_unique) < num_classes or len(pred_unique) < num_classes:
        return True

    if not np.all(np.isfinite(prediction)):
        return True


class Metric(metaclass=__abc__):
    """
    Abstract class for metrics
    """
    @__abs__
    def __str__(self) -> str:
        ...

    @__abs__
    def __call__(
        self, 
        prediction: np.ndarray, 
        y: np.ndarray,
        ) -> np.ndarray:
        ...
        
        
class Accuracy(Metric):
    """
    Class for measuring accuracy metric
    """
    def __init__(self, name = None):
        """
        This calculates the accuracy of the model
        Args:
            name (str, optional): Name of the metric. Defaults to None.
        """
        self.name = 'Accuracy' if name is None else name

    def __str__(self) -> str:
        """
        Returns the name of the metric
        """
        return self.name

    def __call__(self, prediction: ArrayLike, y: ArrayLike):
        prediction = prediction.squeeze()
        y = y.squeeze()
        
        # Compare predictions with true values
        correct = (prediction == y).astype(float)
        
        return __mean__(correct)

class ZeroOneLoss(Metric):
    """
    Class for measuring zero-one loss metric
    """
    def __init__(self, name = None):
        """
        This calculates the zero-one loss of the model
        Args:
            name (str, optional): Name of the metric. Defaults to None.
        """
        self.name = 'ZeroOneLoss' if name is None else name

    def __str__(self) -> str:
        """
        Returns the name of the metric
        """
        return self.name

    def __call__(self, prediction, y):
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values
        y : torch.Tensor
            True values
        """
        from sklearn.metrics import zero_one_loss
        return zero_one_loss(y, prediction)

class F1Score(Metric):
    def __init__(
        self, 
        name=None, 
        average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] | None = "binary",
        sample_weight: ArrayLike | None = None,
        zero_division: int = 0,
        pos_label: int = 1,
        labels: ArrayLike | None = None,
        num_classes: int = 2
        ):
        """
        Computes the F1 score for binary or multi-class classification.
        Args:
            name (str, optional): Name of the metric. Defaults to None.
            average (str, optional): Type of averaging to use. Defaults to 'binary'.
                    binary: Only report results for the class specified by `pos_label`.
                    macro: Calculate metrics for each label, and find their unweighted mean.
                    weighted: Calculate metrics for each label, and find their average weighted by support.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        self.name: str = 'F1Score' if name is None else name
        self.__average: str = average
        self.__num_classes: int = num_classes
        self.__sample_weight: ArrayLike | None = sample_weight
        self.__zero_division: int = zero_division
        self.__pos_label: int = pos_label
        self.__labels: ArrayLike | None = labels

    def __str__(self) -> str:
        return self.name

    def __call__(
        self,
        prediction: torch.Tensor, 
        y: torch.Tensor,
    ):
        # Ensure all unique values are present
        __restrict_if_nunique__(
            y, 
            prediction, 
            num_classes = self.__num_classes
        )
        
        # Compute AUC
        f1 = metrics.f1_score(
            y, 
            prediction,
            average = self.__average,
            sample_weight = self.__sample_weight,
            labels= self.__labels,
            zero_division=self.__zero_division,
            pos_label=self.__pos_label
            )
                        
        # Return as [[value]] array
        return __return_value__(f1)
        
class MatthewsCorrcoef(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MatthewsCorrcoef' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y, prediction, **self.__kwargs)

class Recall(Metric):
    def __init__(
        self, 
        name = None,
        average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] | None = "binary",
        sample_weight: ArrayLike | None = None,
        zero_division: int = 0,
        pos_label: int = 1,
        labels: ArrayLike | None = None,
        num_classes: int = 2
        ):
        """
        Compute recall for binary or multi-class classification.
        Args:
            name (str, optional): Name of the metric. Defaults to None.
            average (str, optional): Type of averaging to use. Defaults to 'binary'.
                    binary: Only report results for the class specified by `pos_label`.
                    macro: Calculate metrics for each label, and find their unweighted mean.
                    weighted: Calculate metrics for each label, and find their average weighted by support.
            num_classes (int, optional): Number of classes. Defaults
        """
        self.__average: str = average
        self.__num_classes: int = num_classes
        self.name: str = 'Recall' if name is None else name
        self.__sample_weight: ArrayLike | None = sample_weight
        self.__zero_division: int = zero_division
        self.__pos_label: int = pos_label
        self.__labels: ArrayLike | None = labels
        
    def __str__(self) -> str:
        return self.name

    def __call__(
        self, 
        prediction: torch.Tensor, 
        y: torch.Tensor,
        ):
        
        # Ensure all unique values are present
        __restrict_if_nunique__(
            y, 
            prediction, 
            num_classes = self.__num_classes
        )
        
        # Compute AUC
        recall = metrics.recall_score(
            y, 
            prediction,
            average = self.__average,
            sample_weight = self.__sample_weight,
            labels= self.__labels,
            zero_division=self.__zero_division,
            pos_label=self.__pos_label
            )
                        
        # Return as [[value]] array
        return __return_value__(recall)
        
        
        

class Jaccard(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Jaccard' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import jaccard_score
        return jaccard_score(prediction, y, **self.__kwargs)

class Precision(Metric):
    def __init__(
        self, 
        name = None,
        average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] | None = "binary",
        sample_weight: ArrayLike | None = None,
        zero_division: int = 0,
        pos_label: int = 1,
        labels: ArrayLike | None = None,
        num_classes: int = 2
        ):
        """
        Compute precision for binary or multi-class classification.
        Args:
            name (str, optional): Name of the metric. Defaults to None.
            average (str, optional): Type of averaging to use. Defaults to 'binary'.
                    binary: Only report results for the class specified by `pos_label`.
                    macro: Calculate metrics for each label, and find their unweighted mean.
                    weighted: Calculate metrics for each label, and find their average weighted by support.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        self.__average = average
        self.__num_classes = num_classes
        self.name = 'Precision' if name is None else name
        self.__sample_weight: ArrayLike | None = sample_weight
        self.__zero_division: int = zero_division
        self.__pos_label: int = pos_label
        self.__labels: ArrayLike | None = labels

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor):
        # Ensure all unique values are present
        __restrict_if_nunique__(
            y, 
            prediction, 
            num_classes = self.__num_classes
        )
        
        # Compute AUC
        recall = metrics.precision_score(
            y, 
            prediction,
            average = self.__average,
            sample_weight = self.__sample_weight,
            labels= self.__labels,
            zero_division=self.__zero_division,
            pos_label=self.__pos_label
            )
                        
        # Return as [[value]] array
        return __return_value__(recall)

class TopKAccuracy(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'TopKAccuracy' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, y, proba):
        # Import libraries
        from .__metrics_handles import handle_probability
        from sklearn.metrics import top_k_accuracy_score

        proba = handle_probability(proba)
        return top_k_accuracy_score(y, proba, **self.__kwargs)

class Auc(Metric):
    def __init__(
        self, 
        name = None, 
        average: Literal['micro', 'macro', 'samples', 'weighted'] | None = "macro",
        sample_weight: ArrayLike | None = None,
        max_fpr: float | None = None,
        multi_class: Literal['raise', 'ovr', 'ovo'] = "raise",
        labels: ArrayLike | None = None,
        num_classes: int = 2
        ):
        self.name: str = 'Auc' if name is None else name
        self.__average: str = average
        self.__sample_weight: ArrayLike | None = sample_weight
        self.__max_fpr: float | None = max_fpr
        self.__multi_class: str = multi_class
        self.__labels: ArrayLike | None = labels
        self.__num_classes = num_classes

    def __str__(self) -> str:
        return self.name
    
    def __call__(
        self, 
        prediction: ArrayLike, 
        y: ArrayLike,
    ):
        # Ensure all unique values are present
        restricted = __restrict_if_nunique__(
            y, 
            prediction, 
            num_classes = self.__num_classes
        )
        
        if restricted:
            return np.array([[0.0]])
        
        # Compute AUC
        auc = metrics.roc_auc_score(
            y, 
            prediction,
            average = self.__average,
            sample_weight = self.__sample_weight,
            max_fpr = self.__max_fpr,
            multi_class = self.__multi_class,
            labels= self.__labels
            )
                        
        # Return as [[value]] array
        return __return_value__(auc)
        


class MeanSquaredError(Metric):
    def __init__(self, name = None,  strategy: str = "mean"):
        """
        Compute the mean squared error (MSE) or squared log error.
        Args:
            name (str, optional): Name of the metric. Defaults to None.
            strategy (str, optional): Strategy to use:
                - 'root': Root mean squared error.
                - 'mean': Mean squared error.
                - 'root_log': Root mean squared log error.
                - 'mean_log': Mean squared log error.
        """
        self.__strategy = strategy
        name = 'Squared' if name is None else name
        match strategy:
            case "root":
                if name[0].isupper():
                    name = "Root" + name + "Error"
                else:
                    name = "root_" + name + "_error"
            case "mean":
                if name[0].isupper():
                    name = "Mean" + name + "Error"
                else:
                    name = "mean_" + name + "_error"
            case "root_log":
                if name[0].isupper():
                    name = "Root" + name + "LogError"
                else:
                    name = "root_" + name + "_log_error"
            case "mean_log":
                if name[0].isupper():
                    name = "Mean" + name + "LogError"
                else:
                    name = "mean_" + name + "_log_error"
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        
        # Convert to tensor and ensure tensors are float
        prediction = torch.tensor(prediction, dtype=torch.float) 
        y =  torch.tensor(y, dtype=torch.float)
        
        mse = torch.nn.functional.mse_loss(prediction, y)
        
        match self.__strategy:
            case "root":
                return torch.sqrt(mse).round(decimals=4)
            case "mean":
                return mse.round(decimals=4)
            case "root_log":
                # Add 1 to prevent log(0) and ensure non-negative values
                log_true = torch.log1p(y.float())
                log_pred = torch.log1p(prediction.float())
                
                # Compute squared log error
                log_error = (log_true - log_pred) ** 2
                
                # Mean and square root
                root_mean_squared_log_e_val = torch.sqrt(torch.mean(log_error))
                return root_mean_squared_log_e_val.round(decimals=4)
            case "mean_log":
                # Add 1 to prevent log(0) and ensure non-negative values
                log_true = torch.log1p(y.float())
                log_pred = torch.log1p(prediction.float())
                
                # Compute squared log error
                log_error = (log_true - log_pred) ** 2
                
                # Mean with log
                mean_log = torch.mean(log_error)
                
                return mean_log.round(decimals=4)
            case _:
                raise ValueError("Invalid strategy")


class MeanAbsoluteError(Metric):
    def __init__(self, name = None):
        """
        Compute the mean absolute error (MAE).
        Args:
            name (str, optional): Name of the metric. Defaults
        """
        self.name = 'MeanAbsoluteError' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        # Convert to tensor and ensure tensors are float
        prediction = torch.tensor(prediction, dtype=torch.float) 
        y =  torch.tensor(y, dtype=torch.float)
        
        mae = torch.nn.functional.l1_loss(prediction, y)
        return mae.round(decimals=4)

class R2(Metric):
    def __init__(self, name = None):
        """
        Compute the R^2 score for regression problem.
        Args:
            name (str, optional): Name of the metric. Defaults
        """
        self.name = 'R2' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        y_true = y.astype(float)
        y_pred = prediction.astype(float)
        
        ss_res = ((y_true - y_pred) ** 2).sum()  # Residual sum of squares
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()  # Total sum of squares
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.round(decimals=4)
