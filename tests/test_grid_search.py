# Praise Ye The Lord

# Import libraries
import sys

sys.path.append(sys.path[0].strip("tests"))

import unittest
from torch import nn
from torch.optim import Adam
from sklearn.datasets import load_wine
from src.exttorch.tuner import GridSearchTune
from src.exttorch.hyperparameter import HyperParameters
from src.exttorch.models import Sequential


def tuned_func(hp: HyperParameters):
    features = hp.Int("features", 1, 3)

    model = Sequential(
        [
            nn.Linear(13, features),
            nn.ReLU(),
            nn.Linear(features, 3),
        ]
    )

    model.compile(loss=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()))

    return model


class TestGridSearch(unittest.TestCase):

    def test_grid_search(self):
        x, y = load_wine(return_X_y=True)

        grid_search = GridSearchTune(
            tuned_func,
        )

        grid_search.search(x, y)


if __name__ == "__main__":
    unittest.main()
