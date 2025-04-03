# Bless Be Name of LORD GOD Of Hosts

# Importing necessary modules
from contexts import *
import torch
from torch import nn
from sklearn.datasets import make_classification
from unittest import TestCase
from exttorch.models import Sequential

def torch_data_generator(batch_size=100):
    while True:
        X, y = make_classification(n_samples=batch_size, n_features=20, n_informative=2, n_classes=2)
        yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def numpy_data_generator():
    while True:
        X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_classes=2)
        yield X, y

class TestAcceptedData(TestCase):
    def setUp(self):                
        self.numpy_data_generator = numpy_data_generator()
        self.torch_data_generator = torch_data_generator()
        
        return super().setUp()
    
    def test_numpy_data(self):
        model = Sequential()
        model.add(nn.Linear(20, 10))
        model.add(nn.ReLU())
        model.add(nn.Linear(10, 2))
        model.add(nn.Sigmoid())
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.numpy_data_generator, epochs=4)
        
    def test_torch_data(self):
        model = Sequential()
        model.add(nn.Linear(20, 10))
        model.add(nn.ReLU())
        model.add(nn.Linear(10, 2))
        model.add(nn.Sigmoid())
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.torch_data_generator, epochs=2)
        
    