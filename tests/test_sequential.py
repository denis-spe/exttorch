# Praise Ye The Lord

# Import libraries
import torch
import unittest as ut
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from exttorch.models import Sequential
from exttorch import models
from sklearn.datasets import load_iris, load_digits
from exttorch.metrics import Accuracy
import pandas as pd
from exttorch._data_handle import DataHandler


class TestSequential(ut.TestCase):
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

        history = self.iris_model.fit(self.ir_x, self.ir_y)

        self.assertIsInstance(history.history, dict)

    def test_model_with_dataloader(self):
        train_data = TensorDataset(torch.tensor(self.ir_x), torch.tensor(self.ir_y))

        train_dataloader = DataLoader(train_data, batch_size=64)

        model = Sequential(
            [
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            ]
        )

        model.compile(
            optimizer=Adam(model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        model.fit(train_dataloader)

    def test_sequential_using_digits_dataset(self):
        digit_model = Sequential(
            [
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ]
        )

        digit_model.compile(
            optimizer=Adam(digit_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=["accuracy"],
        )

        history = digit_model.fit(self.d_x, self.d_y, validation_split=0.5, verbose=1)

        self.assertIsInstance(history.history, dict)

    def test_add_method(self):
        """
        Test the number of layers in
        the model
        """
        digit_model = Sequential([])

        nn.Linear(64, 64)
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(64, 64))
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(64, 10))

        digit_model.compile(
            optimizer=Adam(digit_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=["accuracy"],
        )

        history = digit_model.fit(self.d_x, self.d_y)

    def test_train_model_on_dataFrame(self):
        """
        Test the model on DataFrame
        """

        train_df = pd.read_csv(
            "/Volumes/Storage/DS/DL/exttorch/datasets/digit-recognizer/train.csv"
        )
        test_df = pd.read_csv(
            "/Volumes/Storage/DS/DL/exttorch/datasets/digit-recognizer/test.csv"
        )
        x = train_df.drop("label", axis=1).values
        y = train_df.label.values
        train_dataset = DataHandler(x=x, batch_size=64, y=y)()

        model = Sequential(
            [
                # Transpose Input data
                nn.Flatten(),
                # Input layer
                nn.Linear(in_features=28 * 28, out_features=256),
                nn.ReLU(),  # Activation function
                nn.Dropout(0.4),  # Drop same pixel
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(),  # Activation function
                nn.Dropout(0.4),  # Drop same pixel
                # Output layer
                nn.Linear(in_features=256, out_features=10),
                nn.Softmax(dim=-1),
            ]
        )

        model.compile(
            optimizer=Adam(model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        history = model.fit(
            x, y, epochs=1, batch_size=64, validation_split=0.2, val_batch_size=64
        )

    def test_model_evaluation(self):
        """
        Test the model evaluation
        """
        model = Sequential(
            [
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            ]
        )

        model.compile(
            optimizer=Adam(model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        model.fit(self.ir_x, self.ir_y, random_seed=42)
        eval = model.evaluate(self.ir_x, self.ir_y, random_seed=42)
        self.assertEqual(eval['val_Accuracy'], 0.3333)


if __name__ == "__main__":
    ut.main()
