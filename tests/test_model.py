# Praise be the LORD, for the LORD is good and mercy endures forever

# Import libraries
import unittest as ut
import torch
from torch import nn
from contexts import exttorch
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler
# from exttorch.models import Sequential


# class TestSequential(ut.TestCase):
#     def setUp(self):
#         self.ir_x, self.ir_y = load_iris(return_X_y=True)
#         d_x, self.d_y = load_digits(return_X_y=True)
        
#         scaler = MinMaxScaler()
#         d_x = d_x.reshape(d_x.shape[0], -1)
#         d_x = scaler.fit_transform(d_x)
#         self.d_x = d_x.reshape(-1, 8, 8)
        
#     def test_functional_model(self):
#         x = torch.tensor(self.ir_x, dtype=torch.float32)
#         input_layer = nn.Linear(4, 32)(x)
#         hidden_layer = nn.Linear(32, 32)(input_layer)
#         output_layer = nn.Linear(32, 3)(hidden_layer)
        
#         output_layer(x)
        
        # model = nn.Sequential(input_layer, output_layer)
        
        # model = Model(input_layer, output_layer)
