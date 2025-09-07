# Praise Ye The Lord

# Import libraries
import unittest
from torch import nn
from torch.optim import Adam
from contexts import exttorch
from sklearn.datasets import load_wine
from exttorch.losses import CrossEntropyLoss
from exttorch.optimizers import Adam
from exttorch.tuner import GridSearchTune
from exttorch.hyperparameter import HyperParameters
from exttorch.models import StackedModel


def tuned_func(hp: HyperParameters):
    features = hp.Int("features", 1, 3)

    model = StackedModel(
        [
            nn.Linear(13, features),
            nn.ReLU(),
            nn.Linear(features, 3),
        ]
    )

    model.compile(loss=CrossEntropyLoss(), optimizer=Adam(lr=0.005), metrics=["acc"])

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
