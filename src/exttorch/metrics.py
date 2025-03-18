# Praise Ye The Lord

# Import libraries
from abc import ABCMeta as __abc__, abstractmethod as __abs__
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
        return round((prediction == y).float().mean().item(), 4)

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
    def __init__(self, name=None, average='binary', num_classes: int = 2):
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
        recall_val = Recall()(preds, targets)
        precision_val = Precision()(preds, targets)
        
        if precision_val + recall_val == 0:
            return 0.0
        
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        return round(f1, 4)
    
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
        
        # Compute F1 score for each class
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals)
        
        if average == 'macro':
            return round(f1_scores, 4)
        
        elif average == 'weighted':
            # Weighted by class frequency (support)
            class_support = [(targets == i).sum().float() for i in range(num_classes)]
            total_support = sum(class_support)
            weighted_f1 = sum(f1_scores * torch.tensor(class_support) / total_support)
            return weighted_f1.item()
        else:
            raise ValueError("`average` must be 'macro' or 'weighted'")


    def __call__(self, prediction: torch.Tensor, y: torch.Tensor):
        # Ensure tensors are float
        y_true = y.float()
        y_pred = prediction.float()
        
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
    def __init__(self, name = None, average: str = "binary", num_classes: int = 2):
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
        tp = (preds * targets).sum().item()  # True Positives
        fn = ((1 - preds) * targets).sum().item()  # False Negatives
        
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return round(recall_score, 4)
    
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
            tp = ((preds == class_idx) & (targets == class_idx)).sum().item()
            fn = ((preds != class_idx) & (targets == class_idx)).sum().item()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_scores.append(recall)
        
        return round(sum(recall_scores) / num_classes, 4)  # Macro-average recall

    def __call__(self, prediction, y):
        # Ensure tensors are float
        y_true = y.float()
        y_pred = prediction.float()

        match self.__average:
            case "binary":
                return self.__recall(y_pred, y_true)
            case "macro":
                return self.__multiclass_recall(y_pred, y_true, self.__num_classes)

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
        precision_per_class = []
        
        for c in range(num_classes):
            TP = ((y_pred == c) & (y_true == c)).sum().float()
            FP = ((y_pred == c) & (y_true != c)).sum().float()
            
            class_precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
            precision_per_class.append(class_precision)

        return round(torch.mean(torch.tensor(precision_per_class)).item(), 4)
    
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
    
        # True Positives (TP): Predicted 1, Actual 1
        TP = ((preds == 1) & (targets == 1)).sum().float()

        # False Positives (FP): Predicted 1, Actual 0
        FP = ((preds == 1) & (targets == 0)).sum().float()
    
        # Avoid division by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)

        return round(precision.item(), 4)

    def __call__(self, prediction, y):
        # Ensure tensors are float
        y_true = y.float()
        y_pred = prediction.float()

        match self.__average:
            case "binary":
                return self.__precision(y_pred, y_true)
            case "macro":
                return self.__precision_multiclass(y_pred, y_true, self.__num_classes)

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

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor):
        # Ensure tensors are float
        y_true = y.float()
        y_pred = prediction.float()

        # Sort predictions in descending order
        sorted_indices = torch.argsort(y_pred, descending=True)
        y_true_sorted = y_true[sorted_indices]

        # Count positive and negative samples
        num_positives = torch.sum(y_true)
        num_negatives = y_true.shape[0] - num_positives

        # Compute the cumulative sum of true labels (ranking-based method)
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Compute the true positive rate (TPR) and false positive rate (FPR)
        tpr = tps / num_positives
        fpr = fps / num_negatives

        # Compute the AUC using the trapezoidal rule
        auc = torch.trapz(tpr, fpr)
        
        # Clip the AUC value to be within the range [0, 1]
        auc_value = abs(round(auc.item(), 4))
        auc_value = max(0.0, min(auc_value, 1.0))

        return auc_value

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

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor):
        
        mse = torch.nn.functional.mse_loss(prediction.float(), y.float())
        match self.__strategy:
            case "root":
                return round(torch.sqrt(mse).item(), 4)
            case "mean":
                return round(mse.item(), 4)
            case "root_log":
                # Add 1 to prevent log(0) and ensure non-negative values
                log_true = torch.log1p(y.float())
                log_pred = torch.log1p(prediction.float())
                
                # Compute squared log error
                log_error = (log_true - log_pred) ** 2
                
                # Mean and square root
                root_mean_squared_log_e_val = torch.sqrt(torch.mean(log_error))
                return round(root_mean_squared_log_e_val.item(), 4)
            case "mean_log":
                # Add 1 to prevent log(0) and ensure non-negative values
                log_true = torch.log1p(y.float())
                log_pred = torch.log1p(prediction.float())
                
                # Compute squared log error
                log_error = (log_true - log_pred) ** 2
                
                # Mean with log
                mean_log = torch.mean(log_error)
                
                return round(mean_log.item(), 4)
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

    def __call__(self, prediction, y):
        mae = torch.nn.functional.l1_loss(prediction.float(), y.float())
        return round(mae.item(), 4)

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
        y_true = y.float()
        y_pred = prediction.float()
        
        ss_res = torch.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)  # Total sum of squares
        
        r2 = 1 - (ss_res / ss_tot)
        return round(r2.item(), 4)
