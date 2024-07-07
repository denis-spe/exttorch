# Praise Ye The Lord

# Import libraries
import time
from typing import Any, Callable, Dict, Optional
import itertools as it
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset

from .history import History
from .hyperparameter import HyperParameters
from .model import Sequential
from .__sampler import GridSearchSampler, RandomSearchSampler
from IPython.display import clear_output


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



def change_param_type_to_dict(param_type):
    return {key: value.default
            for key, value in param_type.__dict__.items()}


class BaseSearch:
    def __init__(self, tuned_func,
                objective: str | Callable):
        self.__tuned_func = tuned_func
        self.__obj = objective
        self.each_step_param = {}
        self.__reducing_metric = [
            "mse", "val_mse"
            "MSE", "val_MSE",
            "mae", "val_MSE",
            "MAE", "val_MAE",
            "loss", "val_loss"]
        self.__best_score = 0
        self.__prev_result = 0
        self.best_model = None
        # Best model parameter
        self.__best_param = None
        self.__summary = {}
        self.__best_scores = None

    @property
    def best_scores(self):
        if self.__best_scores is None:
            raise TypeError(
                "First search the parameters with `search` method")

        print(f"{Color.BOLD}Overall Score{Color.END}")
        for key, value in self.__best_scores.items():
            print(f"{key}: {value}")

    @property
    def summary(self):
        if len(self.__summary) == 0:
            raise TypeError(
            "First search the parameters with `search` method")

        reversed_summery = (True
                if (self.__obj not in self.__reducing_metric)
                else False)

        # Sort the summary by objective.
        sorted_summary = dict(sorted(
            self.__summary.items(),
            key=lambda item: item[1][self.__obj],
            reverse=reversed_summery))

        # Print the summary
        print(f"{Color.BOLD}Summary{Color.END}")

        # Loop over the sorted summary
        for key, values in sorted_summary.items():
            print(f"{Color.UNDERLINE}{Color.BOLD}{key}{Color.END}{Color.END}")

            for k, v in values.items():
                if isinstance(v, dict):
                    print(f"{Color.BOLD}{k.title()}{Color.END}")
                    for Key, value in v.items():
                        print(f"{Key}: {value}")
                else:
                    print(f"{Color.BOLD}{k.title()}{Color.END}")
                    print(v)
            print()

    @property
    def best_score(self):
        if self.__best_score is None:
            raise TypeError(
                "First search the parameters with `search` method")

        print(f"{Color.BOLD}Best Score{Color.END}")
        print(f"{self.__obj}: {self.__best_score}")


    @property
    def best_params(self):
        if self.__best_param is None:
            raise TypeError(
                "First search the parameters with `search` method")
        print(f"{Color.BOLD}Best Parameters{Color.END}")
        for key, value in self.__best_param.items():
            print(f"{key}: {value}")

    def tune_summary_table(self, current_params, best_params):
        _current_param = {key: [val] for key, val in current_params.items()}

        if self.__best_param is not None:
            for key, value in best_params.items():
                _current_param[key].append(value)

        data = []

        for items in _current_param.items():
            row = []
            for value in items:
                if isinstance(value, list):
                    for val in value:
                        row.append(val)
                else:
                    row.append(value)
            data.append(row)

        headers = [
            " Param Names ",
            " Previous param ",
            " Best param "
        ]
        from columnar import columnar

        table = columnar(data, headers, no_borders=False, row_sep=" ")
        print(table)



    def __call__(self,
                params,
                iteration,
                n_iterations,
                X, y,
                **kwargs: Any) -> Any:
        # Initializer the tuned function containing the model.
        model = self.__tuned_func(params)

        # Change parameter from class to dictionary like
        # Choice("lr", [0.23, 0.23]) to {'lr': 0.23}
        changed_params = change_param_type_to_dict(params)

        # Add the parameters to the each_step_param dict.
        self.each_step_param[f'step_{iteration + 1}'] = changed_params

        # Print the tuning summary
        print(f"{Color.BOLD}Iteration-{iteration + 1}/{n_iterations}{Color.END}")
        print(f"{Color.BOLD}Summary{Color.END}")
        print(f"| Previous {self.__obj}: {self.__prev_result}")
        print(f"| Best {self.__obj} so far: {self.__best_score}")
        print()

        # Print the tune summary table.
        self.tune_summary_table(changed_params, self.__best_param)
        print()


        print(f"{Color.UNDERLINE} {'-' * 40} {Color.END}")

        # Fit the model
        history = model.fit(X, y=y, **kwargs)

        time.sleep(1)
        clear_output(wait=True)

        # Handle the objective for model evaluation
        self.__handle_objective(model,
                                history=history,
                                iteration=iteration,
                                params = changed_params,
                                )

    def __handle_objective(self,
                        model: Sequential,
                        history: History,
                        iteration: int,
                        params: Dict
                        ):
        """
        Handle objectives

        Args:
            model: (Sequential)
                Fitted Sequential model.
            history: (Dict)
                Model history.
        """
        # Get the metric names from the model.
        #metric_names = (list(map(lambda metric: metric.name, model.metrics))
                #if model.metrics else [])
        #print(metric_names)

        # model history
        model_hist = history.history

        # Get the mean from the metric result
        result = np.mean(model_hist[self.__obj]).round(5)

        self.__summary[f"Iteration {iteration + 1}"] = {
                    self.__obj: result,
                    "parameters": params
                }

        if model.metrics is not None:

            if self.__obj not in self.__reducing_metric:
                if result > self.__best_score:
                    # Assign best_score to best result
                    # from the model.
                    self.__best_score = result
                    # Assign the best model to best_model variable.
                    self.best_model = model
                    # Assign the best model parameters to
                    #  best_params.
                    self.__best_param = params
                    # Store all metrics.
                    self.__best_scores = {
                            key: np.mean(value).round(5)
                            for key, value in model_hist.items()
                        }
            else:
                if self.__best_score == 0 and iteration == 0:
                    # Assign best_score to best result
                    # from the model.
                    self.__best_score = result
                    # Assign the best model to best_model variable.
                    self.best_model = model
                    # Assign the best model parameters to
                    #  best_params.
                    self.__best_param = params

                    # Store all metrics.
                    self.__best_scores = {
                            key: np.mean(value).round(5)
                            for key, value in model_hist.items()
                        }
                else:
                    if result < self.__best_score:
                        # Assign best_score to best result
                        # from the model.
                        self.__best_score = result
                        # Assign the best model to best_model variable.
                        self.best_model = model
                        # Assign the best model parameters to
                        #  best_params.
                        self.__best_param = params

                        # Store all metrics.
                        self.__best_scores = {
                            key: np.mean(value).round(5)
                            for key, value in model_hist.items()
                        }
        else:
            if self.__best_score == 0 and iteration == 0:
                    # Assign best_score to best result
                    # from the model.
                    self.__best_score = result
                    # Assign the best model to best_model variable.
                    self.best_model = model
                    # Assign the best model parameters to
                    # best_params.
                    self.__best_param = params

                    # Store all metrics.
                    self.__best_scores = {
                            key: np.mean(value).round(5)
                            for key, value in model_hist.items()
                        }

            if result < self.__best_score:
                # Assign best_score to best result
                # from the model.
                self.__best_score = result
                # Assign the best model to best_model variable.
                self.best_model = model
                # Assign the best model parameters to
                #  best_params.
                self.__best_param = params

                # Store all metrics.
                self.__best_scores = {
                            key: np.mean(value).round(5)
                            for key, value in model_hist.items()
                        }
        self.__prev_result = result


class GridSearchTune(
    BaseSearch,
    GridSearchSampler):
    def __init__(self,
                tuned_func: Callable,
                objective: str | Callable = "loss"
                ):
        BaseSearch.__init__(self,
                        tuned_func=tuned_func,
                        objective=objective
                        )
        GridSearchSampler.__init__(self)
        self._params = HyperParameters()
        self.__index = 0

    def search(
            self,
            X: np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame,
            y: Optional[np.ndarray | pd.Series |pd.DataFrame] = None,
            **kwargs):
        """
        Search parameters randomly

        Parameters
        ----------
        X: (np.ndarray | DataLoader |
            Dataset | TensorDataset |
            pd.DataFrame)
            Input data for the model.

        y: (Optional[np.ndarray |
            pd.Series |pd.DataFrame])
            Target data for the model.

        kwargs:
            Additional parameters from fit method.

        Examples
        --------
        >>>
        >>> from sklearn.datasets import load_iris
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from exttorch.model import Sequential
        >>> from exttorch.hyperparameter import HyperParameters
        >>> from exttorch.tuner import RandomSearchTune
        >>>
        >>> i_x, i_y = load_iris(return_X_y=True)
        >>> def tuned_model(hp):
        >>>     features = hp.Choice('features', [128, 256, 512, 1062])
        >>>     h_features = hp.Int('h_features', 8, 1062, step=16)
        >>>     lr = hp.Float('lr', 0.0001, 0.001)
        >>>
        >>>     if hp.Boolean('deep_learning'):
        >>>         model = Sequential([
        >>>         nn.Linear(30, features),
        >>>         nn.Linear(features, h_features),
        >>>         nn.Linear(h_features, 2)])
        >>>     else:
        >>>         model = Sequential([
        >>>         nn.Linear(30, features),
        >>>         nn.Linear(features, 2)])
        >>>
        >>>     model.compile(
        >>>         loss = nn.BCEWithLogitsLoss(),
        >>>         optimizer = SGD(model.parameters(), lr=lr),
        >>>         metrics = ["accuracy", "recall"]
        >>>     )
        >>>
        >>>     return model
        >>>
        >>> # Initialize the random search
        >>> random_search = RandomSearchTune(
        >>>                     tuned_model,
        >>>                     objective = 'val_loss'
        >>>                 )
        >>>
        >>> # Search the parameters
        >>> random_search.search(
        >>>                 i_x, i_y,
        >>>                 epochs=5,
        >>>                 validation_data = (i_x, i_y)
        >>>              )
        """

        # Initialize the iterations
        iteration = 0

        # Initialize the index
        index = 0

        # Loop through the iterations
        while True:

            if index == 0:
                # Fit and evaluate the model
                self(self._params,
                    iteration=iteration,
                    n_iterations=self.__index,
                    X=X, y=y, **kwargs)

                # Update the parameters
                self._update_params()

                index = self.product_len

                # Save len of parameters from the product
                self.__index = index

            elif index != 0:
                # Fit and evaluate the model
                self(self._params,
                    iteration=iteration,
                    n_iterations=self.__index,
                    X=X, y=y, **kwargs)

                # Update the parameters
                self._update_params()
                iteration += 1

                if index == iteration:
                    break


class RandomSearchTune(
    BaseSearch,
    RandomSearchSampler
    ):
    def __init__(self,
                tuned_func: Callable,
                random_state: Optional[int] = None,
                objective: str | Callable = "loss",
                iterations: int = 5
            ):
        BaseSearch.__init__(self,
                        tuned_func=tuned_func,
                        objective=objective
                        )
        RandomSearchSampler.__init__(self, random_state=random_state)

        self.__iterations = iterations


    def search(self,
                X: np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame,
                y: Optional[np.ndarray | pd.Series |pd.DataFrame] = None,
                **kwargs):
        """
        Search random parameters

        Parameters:
        -----------
            epochs: (int)
                Number of epochs
            iterations: (int)
                Number of iterations
        """

        # Loop through the number of iterations
        for iteration in range(self.__iterations):
            # Fit and evaluate the model
            self(self._params,
                iteration=iteration,
                n_iterations=self.__iterations,
                 X=X, y=y, **kwargs)

            # Update the parameters
            self._update_params()