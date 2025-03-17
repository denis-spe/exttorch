# Praise Ye The Lord

# Import libraries
import unittest
from contexts import exttorch
from torch import nn
from exttorch.optimizers import Adam
from sklearn.datasets import load_wine
from exttorch.tuner import RandomSearchTune
from exttorch.hyperparameter import HyperParameters
from exttorch.models import Sequential


def tuned_func(hp: HyperParameters):
    features = hp.Int("features", 1, 512)

    model = Sequential(
        [
            nn.Linear(13, features),
            nn.ReLU(),
            nn.Linear(features, 3),
        ]
    )

    model.compile(loss=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()))

    return model


class TestRandomSearch(unittest.TestCase):

    def test_random_search(self):
        x, y = load_wine(return_X_y=True)

        random_search = RandomSearchTune(
            tuned_func,
        )

        random_search.search(x, y)


if __name__ == "__main__":
    unittest.main()
