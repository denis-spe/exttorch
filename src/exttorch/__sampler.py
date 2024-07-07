# Praise Ye The lord

# Import library
from typing import Optional as Optional

class RandomSearchSampler:
    def __init__(self, random_state: Optional[int]):
        from .hyperparameter import HyperParameters
        
        self._params = HyperParameters()
        self._current_param = {}
        self.__random_state = random_state


    def _update_params(self) -> None:
        import numpy as np
        random_state = np.random.RandomState(self.__random_state)

        # Loop over the Parameters
        for key, value in self._params.__dict__.items():
            # Get the new shuffled value.
            new_default = random_state.choice(value.values)

            # Save current parameters.
            self._current_param[key] = new_default

            # Update default to new value.
            self._params.change_default(key, new_default)


class GridSearchSampler:
    def __init__(self):
        from .hyperparameter import HyperParameters
        
        self._params = HyperParameters()
        self._current_param = {}
        self.product = None
        self.product_len = None


    def _update_params(self) -> None:
        import itertools
        
        # Turn HyperParameters into a dict
        hyparam = self._params.__dict__

        # Get the keys
        keys = list(hyparam.keys())

        # Get the values
        values = list(map(lambda x: x.values, hyparam.values()))

        # Get the length of iter product
        self.product_len = len(list(itertools.product(*values)))


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
                self._params.change_default(key, value)

        else:
            # Get the product
            self.product = itertools.product(*values)