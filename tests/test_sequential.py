# Praise Ye The Lord

# Import libraries
import torch
from contexts import exttorch
import unittest as ut
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from exttorch.models import Sequential
from exttorch import models
from sklearn.datasets import load_iris, load_digits
from exttorch.metrics import Accuracy
from exttorch.optimizers import Adam
from exttorch.losses import CrossEntropyLoss, NLLLoss
import pandas as pd
from exttorch.__data_handle import DataHandler
from sklearn.preprocessing import MinMaxScaler


class TestSequential(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        d_x, self.d_y = load_digits(return_X_y=True)
        
        scaler = MinMaxScaler()
        d_x = d_x.reshape(d_x.shape[0], -1)
        d_x = scaler.fit_transform(d_x)
        self.d_x = d_x.reshape(-1, 8, 8)

    def test_sequential_using_iris_dataset(self):
        """
        Test the sequential model using iris dataset
        """
        self.iris_model = Sequential(
            [
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            ]
        )

        self.iris_model.compile(
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        history = self.iris_model.fit(
            self.ir_x, 
            self.ir_y, 
            epochs=2,
            # batch_size=1,
            validation_data = (self.ir_x, self.ir_y),
            verbose="full",
            progress_fill_style="="
            )
        
        print(self.iris_model.predict(self.ir_x))

        # self.assertIsInstance(history.history, dict)

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
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        model.fit(train_dataloader, validation_data=train_dataloader, epochs=5)

    def test_sequential_using_digits_dataset(self):
        digit_model = Sequential(
            [
                nn.Flatten(),
                nn.Linear(64, 1029),
                nn.ReLU(),
                nn.Linear(1029, 1029),
                nn.ReLU(),
                nn.Linear(1029, 10),
            ]
        )

        digit_model.compile(
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
            metrics=["accuracy"],
        )

        history = digit_model.fit(self.d_x, self.d_y, validation_split=0.5)
        
        # digit_model.evaluate(self.d_x, self.d_y, verbose=1)

        # self.assertIsInstance(history.history, dict)

    def test_add_method(self):
        """
        Test the number of layers in
        the model
        """
        digit_model = Sequential([])

        digit_model.add(nn.Flatten())
        digit_model.add(nn.Linear(64, 64))
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(64, 1029))
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(1029, 10))
        # digit_model.add(nn.Softmax(dim=1))

        digit_model.compile(
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
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
            ]
        )

        model.compile(
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        # history = model.fit(
        #     train_dataset, epochs=1
        # )

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
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
            metrics=[Accuracy()],
        )

        model.fit(self.ir_x, self.ir_y, random_seed=42, verbose=None)
        eval = model.evaluate(self.ir_x, self.ir_y, random_seed=42, )
        # self.assertEqual(eval['val_Accuracy'], 0.3333)


if __name__ == "__main__":
    ut.main()
