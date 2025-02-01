# Praise Ye The Lord

"""
======== ExtTorch Tuner ========
"""

# Import libraries
from exttorch._sampler import GridSearchSampler as __Grid__
from exttorch._sampler import RandomSearchSampler as __Random__
from exttorch._tuner_base import BaseSearch


class GridSearchTune(BaseSearch, __Grid__):
    def __init__(self, tuned_func, objective="loss", random_state=None):
        """
        The class represents grid search tune algorithm
        used for hyperparameter tuning the Sequential model
        running the all possible combination of parameters.

        Parameters
        ----------
            tuned_func : (Callable)
                A function containing a tuned sequential model.
            objective : (str | Callable)
                Metric name or metric object for getting the best parameters.
            random_state : (Optional[int]) 
                None by default, for setting seed for reproducibility.
                
        Examples
        --------
        >>> import torch
        >>> from sklearn.datasets import load_iris
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from exttorch.models import Sequential
        >>> from exttorch.hyperparameter import HyperParameters
        >>> from exttorch.tuner import RandomSearchTune
        >>>
        >>>
        >>> i_x, i_y = load_iris(return_X_y=True)
        >>> def tuned_model(hp):
        ...     features = hp.Choice('features', [128, 256, 512, 1062])
        ...     h_features = hp.Int('h_features', 8, 1062, step=16)
        ...     lr = hp.Float('lr', 0.0001, 0.001)
        ...
        ...     if hp.Boolean('deep_learning'):
        ...         model = Sequential([
        ...         nn.Linear(4, features),
        ...         nn.Linear(features, h_features),
        ...         nn.Linear(h_features, 3)])
        ...     else:
        ...         model = Sequential([
        ...         nn.Linear(4, features),
        ...         nn.Linear(features, 3)])
        ...
        ...     model.compile(
        ...         loss = nn.CrossEntropyLoss(),
        ...         optimizer = SGD(model.parameters(), lr=lr),
        ...         metrics = ["accuracy"]
        ...     )
        ...
        ...     return model
        >>>
        >>> # Initialize the random search
        >>> random_search = RandomSearchTune(
        ...    tuned_model,
        ...    objective = 'val_loss',
        ...    random_state=42,
        ... )
        >>>
        >>> # Search the parameters
        >>> random_search.search(i_x, i_y, epochs=5, validation_data = (i_x, i_y)) # doctest: +ELLIPSIS
        \033[1mIteration-1/5\033[0m
        \033[1mSummary\033[0m
        ...
        >>>
        >>> # Best score for random_search
        >>> random_search.best_score # doctest: +ELLIPSIS
        \033[1mBest Score\033[0m
        val_loss: ...
        """
        # Import the HyperParameters
        from .hyperparameter import HyperParameters

        BaseSearch.__init__(self, tuned_func=tuned_func, objective=objective)
        __Grid__.__init__(self)
        self._params = HyperParameters()
        self.__index = 0
        self.__random_state = random_state

    def search(self, X, y=None, **kwargs):
        """
        Searches all possible combination of parameters
        from the tuned function for best parameters.

        Parameters
        ----------
            X : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
                Training feature for training the model.
            y : (Optional[np.ndarray | pd.Series |pd.DataFrame]) None by default,
                Training label for training the model.
            kwargs:
                Sequential model fit method parameters.
        """
        
        # Set the random seed
        if self.__random_state is not None:
            import torch
            torch.manual_seed(self.__random_state)

        # Initialize the iterations
        iteration = 0

        # Initialize the index
        index = 0

        # Loop through the iterations
        while True:

            if index == 0:
                # Fit and evaluate the model
                self(
                    self._params,
                    iteration=iteration,
                    n_iterations=self.__index,
                    X=X,
                    y=y,
                    **kwargs,
                )

                # Update the parameters
                self._update_params()

                index = self.product_len

                # Save len of parameters from the product
                self.__index = index

            elif index != 0:
                # Fit and evaluate the model
                self(
                    self._params,
                    iteration=iteration,
                    n_iterations=self.__index,
                    X=X,
                    y=y,
                    **kwargs,
                )

                # Update the parameters
                self._update_params()
                iteration += 1

                if index == iteration:
                    break


class RandomSearchTune(BaseSearch, __Random__):
    def __init__(self, tuned_func, random_state=None, objective="loss", iterations=5):
        """
        The class represents random search tune algorithm
        used for hyperparameter tuning the Sequential model
        running the random combination of parameters.

        Parameters
        ----------
            tuned_func : (Callable)
                A function containing a tuned sequential model.
            random_state : (Optional[int]) None by default,
                For setting seed for reproducibility.
            objective : (str | Callable) loss by default,
                Metric name or metric object for getting the best parameters.
            iterations : (int) 5 by default,
                Number of iterations for tuning the parameters.
            random_state : (Optional[int]) 
                None by default, for setting seed for reproducibility.
        Examples:
        --------
        >>> import torch
        >>> from sklearn.datasets import load_iris
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from exttorch.models import Sequential
        >>> from exttorch.hyperparameter import HyperParameters
        >>> from exttorch.tuner import RandomSearchTune
        >>>
        >>> i_x, i_y = load_iris(return_X_y=True)
        >>> def tuned_model(hp):
        ...     features = hp.Choice('features', [128, 256, 512, 1062])
        ...     h_features = hp.Int('h_features', 8, 1062, step=16)
        ...     lr = hp.Float('lr', 0.0001, 0.001)
        ...
        ...     if hp.Boolean('deep_learning'):
        ...         model = Sequential([
        ...         nn.Linear(4, features),
        ...         nn.Linear(features, h_features),
        ...         nn.Linear(h_features, 3)])
        ...     else:
        ...         model = Sequential([
        ...         nn.Linear(4, features),
        ...         nn.Linear(features, 3)])
        ... 
        ...     model.compile(
        ...         loss = nn.CrossEntropyLoss(),
        ...         optimizer = SGD(model.parameters(), lr=0.001),
        ...         metrics = ["accuracy"]
        ...     )
        ...
        ...     return model
        >>>
        >>> # Initialize the random search
        >>> random_search = RandomSearchTune(
        ...    tuned_model,
        ...    objective = 'val_loss',
        ...    random_state=42,
        ... )
        >>>
        >>> # Search the parameters
        >>> random_search.search(i_x, i_y, epochs=5, validation_data = (i_x, i_y)) # doctest: +ELLIPSIS
        \033[1mIteration-1/5\033[0m
        \033[1mSummary\033[0m
        ...
        >>>
        >>> # Best score for random_search
        >>> random_search.best_score # doctest: +ELLIPSIS
        \033[1mBest Score\033[0m
        val_loss: ...
        """
        BaseSearch.__init__(self, tuned_func=tuned_func, objective=objective)
        __Random__.__init__(self, random_state=random_state)

        self.__iterations = iterations
        self.__random_state = random_state

    def search(self, X, y=None, **fit_kwargs):
        """
        Searches random combination of parameters
        from the tuned function for the best parameters.

        Parameters
        ----------
            X : (np.ndarray | DataLoader | Dataset | TensorDataset | pd.DataFrame)
                Training feature for training the model.
            y : (Optional[np.ndarray | pd.Series |pd.DataFrame]) None by default,
                Training label for training the model.
            fit_kwargs:
                Sequential fit methods parameters
        """
        
        # Set the random seed
        if self.__random_state is not None:
            import torch
            torch.manual_seed(self.__random_state)

        # Loop through the number of iterations
        for iteration in range(self.__iterations):
            # Fit and evaluate the model
            self(
                self._params,
                iteration=iteration,
                n_iterations=self.__iterations,
                X=X,
                y=y,
                **fit_kwargs,
            )

            # Update the parameters
            self._update_params()
            

if __name__ == "__main__":
    import doctest
    doctest.testmod()