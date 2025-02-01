# Praise The LORD of Host.
"""
======== ExtTorch Tuner Base ========
"""

class BaseSearch:
    def __init__(self, tuned_func, objective):
        self.__tuned_func = tuned_func
        self.__obj = objective
        self.each_step_param = {}
        self.__reducing_metric = [
            "mse",
            "val_mse" "MSE",
            "val_MSE",
            "mae",
            "val_MSE",
            "MAE",
            "val_MAE",
            "loss",
            "val_loss",
        ]
        self.__best_score = 0
        self.__prev_result = 0
        self.best_model = None
        # Best model parameter
        self.__best_param = None
        self.__summary = {}
        self.__best_scores = None

    class __Color:
        PURPLE = "\033[95m"
        CYAN = "\033[96m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        END = "\033[0m"

    @property
    def best_scores(self):
        if self.__best_scores is None:
            raise TypeError("First search the parameters with `search` method")

        print(f"{self.__Color.BOLD}Overall Score{self.__Color.END}")
        for key, value in self.__best_scores.items():
            print(f"{key}: {value}")

    @property
    def summary(self):
        if len(self.__summary) == 0:
            raise TypeError("First search the parameters with `search` method")

        reversed_summery = True if (self.__obj not in self.__reducing_metric) else False

        # Sort the summary by objective.
        sorted_summary = dict(
            sorted(
                self.__summary.items(),
                key=lambda item: item[1][self.__obj],
                reverse=reversed_summery,
            )
        )

        # Print the summary
        print(f"{self.__Color.BOLD}Summary{self.__Color.END}")

        # Loop over the sorted summary
        for key, values in sorted_summary.items():
            print(
                f"{self.__Color.UNDERLINE}{self.__Color.BOLD}{key}{self.__Color.END}{self.__Color.END}"
            )

            for k, v in values.items():
                if isinstance(v, dict):
                    print(f"{self.__Color.BOLD}{k.title()}{self.__Color.END}")
                    for Key, value in v.items():
                        print(f"{Key}: {value}")
                else:
                    print(f"{self.__Color.BOLD}{k.title()}{self.__Color.END}")
                    print(v)
            print()

    @property
    def best_score(self):
        if self.__best_score is None:
            raise TypeError("First search the parameters with `search` method")

        print(f"{self.__Color.BOLD}Best Score{self.__Color.END}")
        print(f"{self.__obj}: {self.__best_score}")

    @property
    def best_params(self):
        if self.__best_param is None:
            raise TypeError("First search the parameters with `search` method")
        print(f"{self.__Color.BOLD}Best Parameters{self.__Color.END}")
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

        headers = [" Param Names ", " Previous param ", " Best param "]
        from columnar import columnar

        table = columnar(data, headers, no_borders=False, row_sep=" ")
        print(table)

    def __call__(self, params, iteration, n_iterations, X, y, **kwargs: any) -> any:

        # Import libraries
        import time
        from IPython.display import clear_output

        # Initializer the tuned function containing the model.
        model = self.__tuned_func(params)

        # Change parameter from class to dictionary like
        # Choice("lr", [0.23, 0.23]) to {'lr': 0.23}
        changed_params = self.__change_param_type_to_dict(params)

        # Add the parameters to the each_step_param dict.
        self.each_step_param[f"step_{iteration + 1}"] = changed_params

        # Print the tuning summary
        print(
            f"{self.__Color.BOLD}Iteration-{iteration + 1}/{n_iterations}{self.__Color.END}"
        )
        print(f"{self.__Color.BOLD}Summary{self.__Color.END}")
        print(f"| Previous {self.__obj}: {self.__prev_result}")
        print(f"| Best {self.__obj} so far: {self.__best_score}")
        print()

        # Print the tune summary table.
        self.tune_summary_table(changed_params, self.__best_param)
        print()

        print(f"{self.__Color.UNDERLINE} {'-' * 40} {self.__Color.END}")

        # Fit the model
        history = model.fit(X, y=y, **kwargs)

        time.sleep(1)
        clear_output(wait=True)

        # Handle the objective for model evaluation
        self.__handle_objective(
            model,
            history=history,
            iteration=iteration,
            params=changed_params,
        )

    @staticmethod
    def __change_param_type_to_dict(param_type):
        return {key: value.default for key, value in param_type.__dict__.items()}

    def __handle_objective(self, model, history, iteration: int, params: dict):
        """
        Handle objectives

        Parameters
        ----------
            model : (Sequential)
                Fitted Sequential model.
            history : (History)
                Model history.
        """
        import numpy as np

        # model history
        model_hist = history.history

        # Get the mean from the metric result
        result = round(np.mean(model_hist[self.__obj]), 5)

        self.__summary[f"Iteration {iteration + 1}"] = {
            self.__obj: result,
            "parameters": params,
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
                        key: round(np.mean(value), 5)
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
                        key: round(np.mean(value), 5)
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
                            key: round(np.mean(value), 5)
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
                    key: round(np.mean(value), 5) for key, value in model_hist.items()
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
                    key: round(np.mean(value), 5) for key, value in model_hist.items()
                }
        self.__prev_result = result