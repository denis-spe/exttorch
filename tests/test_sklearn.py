# May LORD GOD of heaven and earth be glorified forever.

# Import libraries
import torch
from contexts import *
import unittest as ut
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
from exttorch.models import Sequential, Wrapper
from sklearn.pipeline import Pipeline
from exttorch.losses import CrossEntropyLoss
from exttorch.metrics import Accuracy, Recall, F1Score
from exttorch.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class TestPipeline(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        scaler = MinMaxScaler()
        self.ir_x = scaler.fit_transform(self.ir_x)

    def test_pipeline(self):
        """
        Test the pipeline model using iris dataset
        """
        wrapper = Wrapper(
            Sequential(
                [
                    nn.Linear(4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                    nn.Softmax(dim=1),
                ]
            ),
            loss=CrossEntropyLoss(),
            optimizer=Adam(),
            metrics=[Accuracy(), Recall(average='macro')],
            epochs=2,
            validation_split=0.2,
            verbose=None,
        )
        
        self.pipeline_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', wrapper)
        ])
        
        self.pipeline_model.fit(
            self.ir_x, 
            self.ir_y, 
        )
        # print(self.pipeline_model.named_steps['model'].history)
        print(self.pipeline_model.predict(self.ir_x))
    
    def test_cross_validate(self):
        """
        Test the cross-validation of the pipeline model using iris dataset
        """
        wrapper = Wrapper(
            Sequential(
                [
                    nn.Linear(4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                    nn.Softmax(dim=1),
                ]
            ),
            loss=CrossEntropyLoss(),
            optimizer=Adam(),
            metrics=[Accuracy(), Recall(average='macro'), F1Score(average='macro')],
        )
        
        self.pipeline_model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', wrapper)
        ])
        
        scores = cross_validate(wrapper, self.ir_x, self.ir_y, cv=5, params={'epochs': 2})
        print(scores)