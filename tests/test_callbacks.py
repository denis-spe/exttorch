# Praise Ye The Lord

# Import libraries
import sys
import os


import torch
import unittest as ut
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from exttorch.models import Sequential
from exttorch.callbacks import EarlyStopping
from exttorch import models
from sklearn.datasets import load_iris, load_digits
from exttorch.metrics import Accuracy
import pandas as pd
from exttorch._data_handle import DataHandler


class TestCallbacks(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        self.d_x, self.d_y = load_digits(return_X_y=True)

    def test_sequential_using_iris_dataset(self):
        """
        Test the sequential model using iris dataset
        """
        self.iris_model = Sequential(
            [
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            ]
        )

        self.iris_model.compile(
            optimizer=Adam(self.iris_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        history = self.iris_model.fit(
            self.ir_x, self.ir_y,
            callbacks=[EarlyStopping()]
            )
        print(history.history)
        
        self.assertIsInstance(history.history, dict)


if __name__ == "__main__":
    ut.main()
