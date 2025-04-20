# Praise Ye The Lord

# Import libraries
from abc import ABCMeta as __abc__, abstractmethod as __abs__
import numpy as np
import torch

def __mean__(value: torch.Tensor, rounded_by = 4):
    return torch.mean(value, dim=0).round(decimals=rounded_by).view(-1, 1)

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
        prediction: torch.Tensor, 
        y: torch.Tensor,
        device: torch.device = None,
        ) -> torch.Tensor:
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

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor, device: torch.device):
        """
        Parameters
        ----------
        prediction : torch.Tensor
            Predicted values
        y : torch.Tensor
            True values
        """
        prediction = prediction.squeeze()
        y = y.squeeze()
        
        # Compare predictions with true values
        correct = (prediction == y).to(device).float()
        
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
        self.name: str = 'F1Score' if name is None else name
        self.__average: str = average
        self.__num_classes: int = num_classes
        self.__device: torch.device = None

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
        if len(torch.unique(targets)) > 2:
            raise ValueError("F1 score was called with binary average but targets are multiclass")
            
        recall_val = Recall()(preds, targets, self.__device)
        precision_val = Precision()(preds, targets, self.__device)
        
        if precision_val + recall_val == 0:
            return torch.tensor([[0.0]], device=self.__device)
        
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
            -1 by 1 torch.Tensor: F1 score.
        """
        
        # Compute precision and recall for each class
        recall_vals = Recall(average=average, num_classes=num_classes)(preds, targets, self.__device)
        precision_vals = Precision(average=average, num_classes=num_classes)(preds, targets, self.__device)
        
        # Avoid division by zero
        division_part = (precision_vals + recall_vals)
        if division_part == 0:
            return torch.tensor([[0.0]], device=self.__device)
        
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


    def __call__(
        self,
        prediction: torch.Tensor, 
        y: torch.Tensor,
        device: torch.device = None,
    ):
        # Ensure tensors are float
        y_pred = prediction.squeeze().float()
        y_true = y.squeeze().float()
        
        # Set the device.
        self.__device = device
        
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
        self.__average: str = average
        self.__num_classes: int = num_classes
        self.name: str = 'Recall' if name is None else name
        self.__device: torch.device = None

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
            -1 by 1 torch.Tensor: Recall score.
        """
        
        if len(np.unique(targets)) > 2:
            raise ValueError("Recall was called with binary average but targets are multiclass")
                
        tp = (preds * targets).sum()  # True Positives
        fn = ((1 - preds) * targets).sum() # False Negatives
        
        recall_score = (
            (tp / (tp + fn)).round(decimals=4).view(-1, 1) if (tp + fn) > 0 
            else torch.tensor([[0.0]], device=self.__device)
        )
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
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor([[0.0]], device=self.__device)
            recall_scores.append(recall.view(-1, 1))
        
        recall_scores = torch.cat(recall_scores, dim=0).to(self.__device)
        
        return (recall_scores.sum() / num_classes).round(decimals=4).view(-1, 1)  # Macro-average recall

    def __call__(
        self, 
        prediction: torch.Tensor, 
        y: torch.Tensor,
        device: torch.device
        ):
        # Ensure tensors are float
        y_true = y.squeeze().float()
        y_pred = prediction.squeeze().float()
        
        # Set the device.
        self.__device = device

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
        self.__device: torch.device = None
        self.name = 'Precision' if name is None else name

    def __str__(self) -> str:
        return self.name
    
    def __precision_multiclass(self, y_pred, y_true, num_classes) -> torch.Tensor:
        """
        Compute macro-averaged precision for multi-class classification.

        Args:
        y_pred (torch.Tensor): Predicted class indices.
        y_true (torch.Tensor): Ground truth class indices.
        num_classes (int): Number of classes.

        Returns:
            -1 by 1 torch.Tensor: Macro-averaged precision score.
        """
        
        precision_per_class = []
        
        for c in range(num_classes):
            TP = ((y_pred == c) & (y_true == c)).sum().float()
            FP = ((y_pred == c) & (y_true != c)).sum().float()
            
            class_precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor([0.0])
            precision_per_class.append(class_precision.view(-1, 1))
            
        precision_per_class = torch.cat(precision_per_class, dim=0).to(self.__device)
        return __mean__(precision_per_class)
    
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
        if len(torch.unique(targets)) > 2:
            raise ValueError("Precision was called with binary average but targets are multiclass")
        
        # True Positives (TP): Predicted 1, Actual 1
        TP = ((preds == 1) & (targets == 1)).sum().float()

        # False Positives (FP): Predicted 1, Actual 0
        FP = ((preds == 1) & (targets == 0)).sum().float()
    
        # Avoid division by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor([0.0])

        return precision.round(decimals=4).view(-1, 1)  # Reshape to -1 by 1 tensor

    def __call__(self, prediction: torch.Tensor, y: torch.Tensor, device: torch.device):
        # Ensure tensors are float
        y_true = y.squeeze().float()
        y_pred = prediction.squeeze().float()
        
        # Set the device.
        self.__device = device

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
    def __init__(self, name = None, average: str = "binary", num_classes: int = 3):
        """
        Compute the area under the ROC curve (AUC).
        Args:
            name (str, optional): Name of the metric. Defaults to None.
            average (str, optional): Type of averaging to use. Defaults to 'binary'.
                    binary: Only report results for the class specified by `pos_label`.
                    macro: Calculate metrics for each label, and find their unweighted mean.
            num_classes (int, optional): Number of classes. Defaults to 2.
        """
        self.name: str = 'Auc' if name is None else name
        self.__average: str = average
        self.__device: torch.device = None
        self.__num_classes: int = num_classes

    def __str__(self) -> str:
        return self.name
    
    def __call__(
        self, 
        prediction: torch.Tensor, 
        y: torch.Tensor,
        device: torch.device = None,
    ):
        # Ensure tensors are float
        y_true = y.squeeze().float()
        y_pred = prediction.squeeze().float()
        
        # Set the device.
        self.__device = device
        

        match self.__average:
            case "binary":
                if len(torch.unique(y_true)) > 2:
                    raise ValueError("AUC was called with binary average but targets are multiclass")
                return self.__binary_auc(y_true, y_pred)
            case "macro":
                return self.__multiclass_auc(y_true, y_pred)
            case _:
                raise ValueError("`average` must be 'binary' or 'macro'")
                
    
    def __multiclass_auc(self, y_true, y_score):
        labels = y_true.long()    # shape (B,)
        
        if y_score.ndim == 1:
            y_score = y_score.unsqueeze(0)  # shape becomes (1, C)
        
        # If you're doing multiclass, apply softmax:
        y_score = torch.softmax(y_score, dim=1)
                            
        C = self.__num_classes        
        
        auc = []
        for c in range(C):
            # make a binary label: 1 if class==c, else 0
            y_bin = (labels == c).view(-1,1).float().to(self.__device)
            scores_c = y_score[:, c].view(-1,1).float().to(self.__device)
            auc.append(self.__binary_auc(y_bin, scores_c))

        # stack into (C,1)
        return __mean__(torch.cat(auc, dim=0).to(self.__device))

    def __binary_auc(self, probability: torch.Tensor, y: torch.Tensor):
        # 1) Flatten and ensure float
        y_true = y.view(-1).float()
        y_score = probability.view(-1).float()

        # 2) Sort in descending order of score
        sorted_idx = torch.argsort(y_score, descending=True)
        y_sorted = y_true[sorted_idx]

        # 3) Count positives (P) and negatives (N)
        P = y_true.sum().item()
        N = y_true.size(0) - P

        # If only one class present, return 0.0 as tensor
        device, dtype = y.device, y_score.dtype
        if P == 0 or N == 0:
            return torch.tensor([[0.0]], device=device, dtype=dtype)

        # 4) Cumulative true positives and false positives
        tps = torch.cumsum(y_sorted, dim=0)
        fps = torch.cumsum(1 - y_sorted, dim=0)

        # 5) TPR and FPR
        tpr = tps / P
        fpr = fps / N

        # 6) Pad with (0,0) and (1,1)
        zero = torch.tensor([0.], device=device, dtype=dtype)
        one  = torch.tensor([1.], device=device, dtype=dtype)
        tpr = torch.cat([zero, tpr, one])
        fpr = torch.cat([zero, fpr, one])

        # 7) Compute AUC via trapezoidal rule
        auc_val = torch.trapz(tpr, fpr).item()

        # 8) Clip to [0,1] and round to 4 decimals
        auc_clipped = round(max(0.0, min(1.0, auc_val)), 4)

        # Return as a (1,1) tensor
        return torch.tensor([[auc_clipped]], device=device, dtype=dtype)


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
