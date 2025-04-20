""" Praise Ye The Lord God Almighty """

# Import Libraries
from typing import Any, Dict, List


class History:
    """
    Represents model history.
    """

    def __init__(self, metrics: Any) -> None:
        """
        Represents model history.

        Parameters
        ----------
        metrics : Any
            List of metrics
        """

        # Get the metric names
        names = [metric if type(metric) == str else str(metric) for metric in metrics]

        # Create the history
        self.__history = {name: [] for name in names}

        # Create the validation history
        self.__history.update({"val_" + name: [] for name in names})

        # Create the loss list in history
        self.__history["loss"] = []
        self.__history["val_loss"] = []

    @property
    def history(self) -> Dict:
        """
        Returns the model history
        """
        return {key: value for key, value in self.__history.items() if len(value) > 0}

    def add_history(self, metric: Dict[str, float]) -> None:
        # Loop over the key and value from metric
        for key, value in metric.items():

            # Append the result to the history
            self.__history[key].append(value)
