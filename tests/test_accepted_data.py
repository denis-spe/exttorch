# Bless Be Name of LORD GOD Of Hosts

# Importing necessary modules
from contexts import *
import torch
from torch import nn
from sklearn.datasets import make_classification, make_regression
from unittest import TestCase
from exttorch.models import Sequential
from exttorch.metrics import Precision, Recall, F1Score, Auc

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
        model.add(nn.Linear(10, 1))
        model.add(nn.Sigmoid())
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.numpy_data_generator, epochs=4)
        
    def test_torch_data(self):
        model = Sequential()
        model.add(nn.Linear(20, 10))
        model.add(nn.ReLU())
        model.add(nn.Linear(10, 1))
        model.add(nn.Sigmoid())
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.torch_data_generator, epochs=2)
        
    def test_regression_dataset(self):
        
        X, y = make_regression(n_samples=250, n_features=20, noise=0.1)
        
        model = Sequential()
        model.add(nn.Linear(20, 10))
        model.add(nn.ReLU())
        model.add(nn.Linear(10, 1))
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=4, batch_size=32, validation_split=0.2)
        
    def test_binary_classification_dataset(self):
        
        X, y = make_classification(n_samples=200, n_features=20, n_informative=2, n_classes=2)
        
        model = Sequential()
        model.add(nn.Linear(20, 256))
        model.add(nn.ReLU())
        model.add(nn.Linear(256, 512))
        model.add(nn.ReLU())
        model.add(nn.Linear(512, 1029))
        model.add(nn.ReLU())
        model.add(nn.Linear(1029, 1029))
        model.add(nn.ReLU())
        model.add(nn.Linear(1029, 1))
        model.add(nn.Sigmoid())
        
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['auc']#, 'precision', 'recall', 'f1_score']
            )
        model.fit(X, y, epochs=1, batch_size=16)
        
    def test_multiclass_classification_dataset(self):
        
        n_label = 3
        n_sample = 300
        
        X, y = make_classification(n_samples=n_sample, n_features=20, n_informative=9, n_classes=n_label)
        
        model = Sequential()
        model.add(nn.Linear(20, 256))
        model.add(nn.ReLU())
        model.add(nn.Linear(256, 512))
        model.add(nn.ReLU())
        model.add(nn.Linear(512, 1029))
        model.add(nn.ReLU())
        model.add(nn.Linear(1029, 1029))
        model.add(nn.ReLU())
        model.add(nn.Linear(1029, n_label))
        
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=[
                Auc(multi_class="ovr", num_classes=n_label), 
                # Auc(average='macro', num_classes=n_label), 
                # F1Score(average='macro', num_classes=n_label), 
                # Recall(average='macro', num_classes=n_label), 
                # Precision(average='macro', num_classes=n_label)
                ]
            )
        model.fit(X, y, epochs=1, batch_size=16)