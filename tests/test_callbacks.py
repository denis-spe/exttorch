# Praise Ye The Lord

# Import libraries
import unittest as ut

from sklearn.datasets import load_iris, load_digits
from torch import nn

from src.exttorch import callbacks
from src.exttorch.callbacks import EarlyStopping
from src.exttorch.models import StackedModel


class TestCallbacks(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        self.d_x, self.d_y = load_digits(return_X_y=True)

    def test_sequential_using_iris_dataset(self):
        """
        Test the sequential model using iris dataset
        """
        self.iris_model = StackedModel(
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
            self.ir_x,
            self.ir_y,
            validation_data=(self.ir_x, self.ir_y),
            epochs=10,
            callbacks=[EarlyStopping(patience=3, monitor="val_acc")],
        )

        self.assertIsInstance(history.history, dict)

    def test_is_there_check_point(self):
        self.assertTrue(hasattr(callbacks, "SaveOnCheckpoint"))

    def test_check_point_instance(self):
        try:
            callbacks.SaveOnCheckpoint(
                ".",
                monitor="val_loss",
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
            )
        except AttributeError:
            self.fail("Invalid parameters")

    def test_sequential_using_SaveOnCheckpoint(self):
        """
        Test the sequential model using iris dataset
        """
        self.iris_model = StackedModel(
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

        checkpoint = callbacks.SaveOnCheckpoint(
            "./tmp/checkpoint.json",
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        self.iris_model.fit(
            self.ir_x,
            self.ir_y,
            validation_data=[self.ir_x, self.ir_y],
            epochs=1,
            callbacks=[checkpoint],
        )


if __name__ == "__main__":
    ut.main()
