# Praise Ye The Lord

# Import libraries
import time
from typing import Any, Callable, Dict, Optional
import itertools as it
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from exttorch.history import History
from exttorch.hyperparameter import HyperParams
from exttorch.model import Sequential
from IPython.display import clear_output


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

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

        reversed = (True
                if (self.__obj not in self.__reducing_metric)
                else False)

        # Sort the summary by objective.
        sorted_summary = dict(sorted(
            self.__summary.items(),
            key=lambda item: item[1][self.__obj],
            reverse=reversed))

        # Print the summary
        print(f"{Color.BOLD}Summary{Color.END}")

        # Loop over the sorted summary
        for key, values in sorted_summary.items():
            print(f"{Color.UNDERLINE}{Color.BOLD}{key}{Color.END}{Color.END}")

            for k, v in values.items():
                if isinstance(v, dict):
                    print(f"{Color.BOLD}{k.title()}{Color.END}")
                    for key, value in v.items():
                        print(f"{key}: {value}")
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
                 **kwds: Any) -> Any:
        # Initializer the tuned function containing the model.
        model = self.__tuned_func(params)

        # Change parameter from class to dictionary like
        # Choice("lr", [0.23, 0.23]) to {'lr': 0.23}
        changed_params = self.change_param_type_to_dict(params)

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
        history = model.fit(X, y=y, **kwds)

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
                    # to best_params.
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
                    # to best_params.
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
                        # to best_params.
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
                    # to best_params.
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
                # to best_params.
                self.__best_param = params

                # Store all metrics.
                self.__best_scores = {
                             key: np.mean(value).round(5)
                             for key, value in model_hist.items()
                         }
        self.__prev_result = result



    def change_param_type_to_dict(self, param_type):
         return {key: value.default
                 for key, value in param_type.__dict__.items()}

class RandomSearchSampler:
    def __init__(self, random_state: Optional[int]):
        self._params = HyperParams()
        self._current_param = {}
        self.__random_state = random_state


    @property
    def _update_params(self) -> None:
        random_state = np.random.RandomState(self.__random_state)

        # Loop over the Parameters
        for key, value in self._params.__dict__.items():
            # Get the new shuffled value.
            new_default = random_state.choice(value.values)

            # Save current parameters.
            self._current_param[key] = new_default

            # Update default to new value.
            self._params._change_default(key, new_default)


class GridSearchSampler:
    def __init__(self):
        self._params = HyperParams()
        self._current_param = {}
        self.product = None
        self.product_len = None


    @property
    def _update_params(self) -> None:
        # Turn HyperParams into a dict
        hyparam = self._params.__dict__

        # Get the keys
        keys = list(hyparam.keys())

        # Get the values
        values = list(map(lambda x: x.values, hyparam.values()))

        # Get the length of iter product
        self.product_len = len(list(it.product(*values)))


        if self.product is not None:
            # Get the length of the product
            # print(list(self.product))

            # Get the next product
            next_product = next(self.product)

            params = { key: value
                   for key, value in zip(keys, next_product)}

            # Update default to new value.
            for key, value in params.items():
                # Save current parameters.
                self._current_param[key] = value

                # Update default to new value.
                self._params._change_default(key, value)

        else:
            # Get the product
            self.product = it.product(*values)


class GridSearchTune(
    BaseSearch,
    GridSearchSampler):
    def __init__(self,
                 tuned_func: Callable,
                 random_state: Optional[int] = None,
                 objective: str | Callable = "loss"
                 ):
        BaseSearch.__init__(self,
                        tuned_func=tuned_func,
                        objective=objective
                        )
        GridSearchSampler.__init__(self)
        self._params = HyperParams()
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
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>>
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
        >>>                 X, y,
        >>>                 epochs=5,
        >>>                 validation_data = (b_x, b_y)
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
                self._update_params

                index = self.product_len

                # Save len of parameters from the product
                self.__index = index

            elif index != 0:
                # Fit and evalate the model
                self(self._params,
                     iteration=iteration,
                     n_iterations=self.__index,
                     X=X, y=y, **kwargs)

                # Update the parameters
                self._update_params
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
            # Fit and evalate the model
            self(self._params,
                iteration=iteration,
                n_iterations=self.__iterations,
                 X=X, y=y, **kwargs)

            # Update the parameters
            self._update_params