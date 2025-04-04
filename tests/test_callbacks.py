# Praise Ye The Lord

# Import libraries
import unittest as ut
from contexts import exttorch
from exttorch.callbacks import EarlyStopping
from exttorch.metrics import Accuracy
from exttorch.models import Sequential
from sklearn.datasets import load_iris, load_digits
from torch import nn
from torch.optim import Adam


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
            optimizer="adam",
            loss="crossentropy",
            metrics=["acc"],
        )

        history = self.iris_model.fit(
            self.ir_x, self.ir_y,
            validation_data=[self.ir_x, self.ir_y],
            epochs=120,
            callbacks=[EarlyStopping(patience=2)]
            )
        
        self.assertIsInstance(history.history, dict)


if __name__ == "__main__":
    ut.main()
