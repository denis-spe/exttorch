# import sys
# sys.path.append(sys.path[0].replace("tests", "src"))

import unittest as ut
from torch import nn
from torch.optim import Adam
from src.exttorch.model import Sequential
from sklearn.datasets import load_iris, load_digits

class TestSequential(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        self.d_x, self.d_y = load_digits(return_X_y=True)

    def test_sequential_using_iris_dataset(self):
        self.iris_model = Sequential([
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ])

        self.iris_model.compile(
            optimizer=Adam(self.iris_model.parameters()),
            loss=nn.CrossEntropyLoss()
        )
        history = self.iris_model.fit(self.ir_x, self.ir_y)
        
        self.assertIsInstance(history.history, dict)
        
    def test_sequential_using_digits_dataset(self):
        digit_model = Sequential([
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ])
        
        digit_model.compile(
            optimizer=Adam(digit_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=['accuracy']
        )
        
        history = digit_model.fit(self.d_x, self.d_y, validation_split=0.5, verbose=1)
        
        self.assertIsInstance(history.history, dict)

    def test_add_method(self):
        """
        Test the number of layers in
        the model
        """
        digit_model = Sequential([
            nn.Flatten(),
            nn.Linear(64, 10)
        ])

        digit_model.add(nn.Linear(64, 64))
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(64, 64))
        digit_model.add(nn.ReLU())
        digit_model.add(nn.Linear(64, 10))
        # self.assertEqual(len(digit_model.layers), 2)

        digit_model.compile(
            optimizer=Adam(digit_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=['accuracy']
        )

        history = digit_model.fit(self.d_x, self.d_y)



if __name__ == '__main__':
    ut.main()