import sys
sys.path.append(sys.path[0].replace("tests", "src"))

import unittest as ut
from torch import nn
from torch.optim import Adam
from exttorch.model import Sequential
from sklearn.datasets import load_iris, load_digits

class TestSequential(ut.TestCase):
    def setUp(self):
        self.iris_model = Sequential([
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        ])
        
        self.digit_model = Sequential([
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ])

    def test_sequential_using_iris_dataset(self):
        x, y = load_iris(return_X_y=True)
        
        self.iris_model.compile(
            optimizer=Adam(self.iris_model.parameters()),
            loss=nn.CrossEntropyLoss()
        )
        history = self.iris_model.fit(x, y)
        
        self.assertIsInstance(history.history, dict)
        
    def test_sequential_using_digits_dataset(self):
        x, y = load_digits(return_X_y=True)
        
        print(x.shape)
        
        self.digit_model.compile(
            optimizer=Adam(self.digit_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=['accuracy']
        )
        
        history = self.digit_model.fit(x, y)
        
        self.assertIsInstance(history.history, dict)

if __name__ == '__main__':
    ut.main()