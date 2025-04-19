# Praise Ye The Lord

# Import libraries
from abc import ABCMeta as __abc__, abstractmethod as __abs__
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

class Metric(metaclass=__abc__):
    """
    Abstract class for metrics
    """
    @__abs__
    def __str__(self) -> str:
        ...

    @__abs__
    def __call__(self, prediction: torch.Tensor, y: torch.Tensor, size: int):
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

    def __call__(self, prediction, y):
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values
        y : torch.Tensor
            True values
        """
        return (prediction == y).astype(float).mean().round(decimals=4)

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
    def __init__(self, name=None, average='binary', num_classes: int = 3, zero_division='warn'):
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
        self.name = 'F1Score' if name is None else name
        self.__average = average
        self.__num_classes = num_classes

    def __str__(self) -> str:
        return self.name
    
    def __f1_score(self, preds, targets):
        """
        Compute F1 score for binary classification.
        
        Args:
        preds (torch.Tensor): Model predictions (probabilities or logits).
        targets (torch.Tensor): Ground truth labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.

        Returns:
        float: F1 score.
        """
        if len(np.unique(targets)) > 2:
            raise ValueError("F1 score was called with binary average but targets are multiclass")
            
        recall_val = Recall()(preds, targets)
        precision_val = Precision()(preds, targets)
        
        if precision_val + recall_val == 0:
            return 0.0
        
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        return f1.round(decimals=4)
    
    def __f1_score_multiclass(self, preds, targets, num_classes, average='macro'):
        """
        Compute F1 score for multi-class classification.
        
        Args:
        preds (torch.Tensor): Model predictions (probabilities or logits).
        targets (torch.Tensor): Ground truth labels (0 to num_classes-1).
        num_classes (int): Number of classes.
        average (str): How to average the scores: 'macro' or 'weighted'.
        threshold (float): Threshold to convert probabilities to binary predictions.
        
        Returns:
        float: F1 score.
        """
        
        # Compute precision and recall for each class
        recall_vals = Recall(average=average, num_classes=num_classes)(preds, targets)
        precision_vals = Precision(average=average, num_classes=num_classes)(preds, targets)
        
        # Avoid division by zero
        division_part = (precision_vals + recall_vals)
        if division_part == 0:
            return 0.0
        
        # Compute F1 score for each class
        f1_scores = 2 * (precision_vals * recall_vals) / division_part
        
        if average == 'macro':
            return f1_scores.round(decimals=4)
        
        elif average == 'weighted':
            # Weighted by class frequency (support)
            class_support = [(targets == i).sum().astype(float) for i in range(num_classes)]
            total_support = sum(class_support)
            weighted_f1 = sum(f1_scores * np.array(class_support) / total_support)
            return weighted_f1.round(decimals=4)
        else:
            raise ValueError("`average` must be 'macro' or 'weighted'")


    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        # Ensure tensors are float
        y_true = y.astype(float)
        y_pred = prediction.astype(float)
        
        match self.__average:
            case "binary":
                return self.__f1_score(y_pred, y_true)
            case "macro":
                return self.__f1_score_multiclass(y_pred, y_true, self.__num_classes, average='macro')
            case "weighted":
                return self.__f1_score_multiclass(y_pred, y_true, self.__num_classes, average='weighted')
            case _:
                raise ValueError("`average` must be 'macro' or 'weighted'")
        
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
    def __init__(self, name = None, average: str = "binary", num_classes: int = 3):
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
        self.__average = average
        self.__num_classes = num_classes
        self.name = 'Recall' if name is None else name

    def __str__(self) -> str:
        return self.name
    
    def __recall(self, preds, targets):
        """
        Compute recall for binary classification.
        
        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
            targets (torch.Tensor): Ground truth labels (0 or 1).
            threshold (float): Decision threshold for converting probabilities to binary values.
        
        Returns:
            float: Recall score.
        """
        
        if len(np.unique(targets)) > 2:
            raise ValueError("Recall was called with binary average but targets are multiclass")
                
        tp = (preds * targets).sum()  # True Positives
        fn = ((1 - preds) * targets).sum() # False Negatives
        
        recall_score = (tp / (tp + fn)).round(decimals=4) if (tp + fn) > 0 else 0.0
        return recall_score
    
    def __multiclass_recall(self, preds, targets, num_classes):
        """
        Compute macro-average recall for multi-class classification.
        
        Args:
            preds (torch.Tensor): Model predictions (logits or probabilities).
            targets (torch.Tensor): Ground truth labels.
            num_classes (int): Number of classes.
        
        Returns:
            float: Macro-averaged recall.
        """
        recall_scores = []
        
        for class_idx in range(num_classes):
            tp = ((preds == class_idx) & (targets == class_idx)).sum()
            fn = ((preds != class_idx) & (targets == class_idx)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_scores.append(recall)
        
        return (sum(recall_scores) / num_classes).round(decimals=4)  # Macro-average recall

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        # Ensure tensors are float
        y_true = y.astype(float)
        y_pred = prediction.astype(float)

        match self.__average:
            case "binary":
                return self.__recall(y_pred, y_true)
            case "macro":
                return self.__multiclass_recall(y_pred, y_true, self.__num_classes)                
            case "weighted":
                return self.__multiclass_recall(y_pred, y_true, self.__num_classes)
            case _:
                raise ValueError("`average` must be 'macro' or 'weighted'")

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
    def __init__(self, name = None,  average: str = "binary", num_classes: int = 2):
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

    def __str__(self) -> str:
        return self.name
    
    def __precision_multiclass(self, y_pred, y_true, num_classes):
        """
        Compute macro-averaged precision for multi-class classification.

        Args:
        y_pred (torch.Tensor): Predicted class indices.
        y_true (torch.Tensor): Ground truth class indices.
        num_classes (int): Number of classes.

        Returns:
        float: Macro-averaged precision score.
        """
        
        precision_per_class = np.array([])
        
        for c in range(num_classes):
            TP = ((y_pred == c) & (y_true == c)).sum().astype(float)
            FP = ((y_pred == c) & (y_true != c)).sum().astype(float)
            
            class_precision = TP / (TP + FP) if (TP + FP) > 0 else np.array([0.0])
            precision_per_class = np.append(precision_per_class, class_precision)

        return precision_per_class.mean().round(decimals=4)
    
    def __precision(self, preds, targets):
        """
        Compute precision for binary classification.
        
        Args:
        preds (torch.Tensor): Model predictions (probabilities or logits).
        targets (torch.Tensor): Ground truth labels (0 or 1).
        threshold (float): Threshold to convert probabilities to binary predictions.

        Returns:
        float: Precision score.
        """
        if len(np.unique(targets)) > 2:
            raise ValueError("Precision was called with binary average but targets are multiclass")
        
        # True Positives (TP): Predicted 1, Actual 1
        TP = ((preds == 1) & (targets == 1)).sum().astype(float)

        # False Positives (FP): Predicted 1, Actual 0
        FP = ((preds == 1) & (targets == 0)).sum().astype(float)
    
        # Avoid division by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else np.array([0.0])

        return precision.round(decimals=4)

    def __call__(self, prediction: np.ndarray, y: np.ndarray):
        # Ensure tensors are float
        y_true = y.astype(float)
        y_pred = prediction.astype(float)

        match self.__average:
            case "binary":
                return self.__precision(y_pred, y_true)
            case "macro":
                return self.__precision_multiclass(y_pred, y_true, self.__num_classes)
            case "weighted":
                return self.__precision_multiclass(y_pred, y_true, self.__num_classes)
            case _:
                raise ValueError("`average` must be 'macro' or 'weighted'")

class TopKAccuracy(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'TopKAccuracy' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, y, proba):
        # Import libraries
        from ._metrics_handles import handle_probability
        from sklearn.metrics import top_k_accuracy_score

        proba = handle_probability(proba)
        return top_k_accuracy_score(y, proba, **self.__kwargs)

class Auc(Metric):
    def __init__(self, name = None):
        """
        Compute the area under the ROC curve (AUC).
        Args:
            name (str, optional): Name of the metric. Defaults to None.
        """
        self.name = 'Auc' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, probability: np.ndarray, y: np.ndarray):
        # 1) Ensure float
        y_true = y.astype(float)
        y_pred = probability.astype(float)
        
        y_true   = np.asarray(y_true,   dtype=float).ravel()  # shape (N,)
        y_pred  = np.asarray(y_pred,  dtype=float).ravel()  # shape (N,)
                
        # 2) Sort in descending order
        idx = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[idx]

        # 3) Count positives/negatives
        P = y_true.sum()
        N = y_true.shape[0] - P
        
        # If either class is missing, return 0.0 (or np.nan, as you prefer)
        if P == 0 or N == 0:
            return 0.0

        # 4) Cumulative true/false positives
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        # 5) TPR and FPR
        tpr = tps / P
        fpr = fps / N

        # 6) Pad with (0,0) and (1,1) for a complete ROC curve
        tpr = np.concatenate([[0.], tpr, [1.]])
        fpr = np.concatenate([[0.], fpr, [1.]])

        # 7) Compute AUC
        auc = np.trapz(tpr, fpr)

        # 8) Clip & round
        auc = float(np.clip(auc, 0.0, 1.0).round(4))
        return auc


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
