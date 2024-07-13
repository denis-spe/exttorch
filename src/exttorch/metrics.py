# Praise Ye The Lord

# Import libraries
from abc import ABCMeta as __abc__, abstractmethod as __abs__

class Metric(metaclass=__abc__):
    """
    Abstract class for metrics
    """
    @__abs__
    def __str__(self) -> str:
        ...

    @__abs__
    def __call__(self, prediction, y):
        ...
        


class Accuracy(Metric):
    """
    Class for measuring accuracy metric
    """
    def __init__(self, name = None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the metric, by default None
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
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, prediction)


class ZeroOneLoss(Metric):
    """
    Class for measuring zero-one loss metric
    """
    def __init__(self, name = None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the metric, by default None
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
    def __init__(self, name = None, **kwargs) -> None:
        self.__kwargs = kwargs
        self.name = 'F1Score' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import f1_score
        return f1_score(y, prediction, **self.__kwargs)

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
    def __init__(self, name = None,  **kwargs):
        self.__kwargs = kwargs
        self.name = 'Recall' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import recall_score
        return recall_score(prediction, y, **self.__kwargs)

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
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Precision' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import precision_score
        return precision_score(prediction, y, **self.__kwargs)

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
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'Auc' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, y, proba):
        # Import libraries
        from ._metrics_handles import handle_probability
        from sklearn.metrics import roc_auc_score

        proba = handle_probability(proba)
        return roc_auc_score(y, proba, **self.__kwargs)


class MeanSquaredError(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MeanSquaredError' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(prediction, y.reshape(y.shape[0], 1))


class MeanAbsoluteError(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'MeanAbsoluteError' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(prediction, y.reshape(y.shape[0], 1),
                                   **self.__kwargs)

class R2(Metric):
    def __init__(self, name = None, **kwargs):
        self.__kwargs = kwargs
        self.name = 'R2' if name is None else name

    def __str__(self) -> str:
        return self.name

    def __call__(self, prediction, y):
        from sklearn.metrics import r2_score
        return r2_score(
            prediction.view(prediction.shape[0], 1),
            y.reshape(y.shape[0], 1),
            **self)
