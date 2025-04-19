# Praise Ye The Lord

# Import libraries
import torch
import numpy as np
from contexts import exttorch
import unittest as ut
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from exttorch.models import Sequential
from exttorch import models
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.datasets import load_digits, load_wine, load_iris
from exttorch.metrics import (
    Accuracy, 
    MeanSquaredError, 
    MeanAbsoluteError, 
    R2, 
    Auc,
    F1Score,
    Recall,
    Precision
    )

import pandas as pd
from exttorch._data_handle import DataHandler

class TestMetrics(ut.TestCase):
    def setUp(self):
        self.ir_x, self.ir_y = load_iris(return_X_y=True)
        self.d_x, self.d_y = load_digits(return_X_y=True)
        self.w_x, self.w_y = load_wine(return_X_y=True)
        self.cancer_x, self.cancer_y = load_breast_cancer(return_X_y=True)
        self.clf = LogisticRegression(solver="liblinear", random_state=0).fit(self.cancer_x, self.cancer_y)
        self.cancer_pred = self.clf.predict(self.cancer_x)
        self.cancer_prob = self.clf.predict_proba(self.cancer_x)[:, 1]
        
    def test_accuracy_metric(self):
        """
        Test the accuracy metric
        """
        accuracy = Accuracy()
        acc = accuracy(
            np.array([1, 0, 1]), 
            np.array([1, 1, 1])
            )
        self.assertEqual(acc, 0.6667)
        
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 2, 0])
        acc = accuracy(y_pred, y_true)
        self.assertEqual(acc, 0.25)
        
        
    def test_recall_metric(self):
        """
        Test the recall metric
        """
        recall = Recall(average="binary")
        rec = recall(np.array([1, 0, 1]), np.array([1, 1, 1]))
        self.assertEqual(rec, 0.6667)
        
        recall = Recall(average="macro", num_classes=3)
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 2, 0])
        rec = recall(y_pred, y_true)
        self.assertEqual(rec, 0.3333)
        
        test_case = recall_score(self.cancer_y, self.cancer_pred)
    
        recall = Recall()
        y_true = self.cancer_y
        y_pred = self.cancer_pred
        recall_val = recall(y_pred, y_true)
        self.assertEqual(recall_val, round(test_case, 4))
        
    def test_precision_metric(self):
        """
        Test the precision metric
        """
        precision = Precision(average="binary")
        prec = precision(np.array([1, 0, 1]), np.array([1, 1, 1]))
        self.assertEqual(prec, 1.0)
        
        precision = Precision(average="macro", num_classes=3)
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 2, 0])
        prec = precision(y_pred, y_true)
        self.assertEqual(prec, 0.1667)
        
    def test_f1_score_metric(self):
        """
        Test the f1 score metric
        """
        f1 = F1Score(average="binary")
        f1_val = f1(np.array([1, 0, 1]), np.array([1, 1, 1]))
        self.assertEqual(f1_val, 0.8)
        
        f1 = F1Score(average="macro", num_classes=3)
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1])
        y_pred = np.array([0, 2, 1, 0, 0, 1, 2, 0])
        f1_val = f1(y_pred, y_true)
        self.assertEqual(f1_val, 0.2222)
        
        f1 = F1Score(average="weighted", num_classes=3)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 1])
        f1_val = f1(y_pred, y_true)
        self.assertEqual(f1_val, 0.2666)
    
    
        
    def test_mean_squared_error_metric(self):
        """
        Test the mean squared error metric
        """
        mse = MeanSquaredError()
        mse_val = mse(np.array([1, 0, 1]), np.array([1, 1, 1]))
        self.assertEqual(mse_val, 0.3333)
        
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mse_val = mse(y_pred, y_true)
        self.assertEqual(mse_val, 0.375)
        
        log_mse = MeanSquaredError(strategy="mean_log")
        y_true = np.array([3, 5, 2.5, 7])
        y_pred = np.array([2.5, 5, 4, 8])
        mse_val = log_mse(y_pred, y_true)
        self.assertEqual(mse_val, 0.0397)
        
        root_mse = MeanSquaredError(strategy="root")
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mse_val = root_mse(y_pred, y_true)
        self.assertEqual(mse_val, 0.6124)
        
        root_log_mse = MeanSquaredError(strategy="root_log")
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mse_val = root_log_mse(y_pred, y_true)
        self.assertEqual(mse_val,  0.3578)
        
        y_true = np.array([3, 5, 2.5, 7])
        y_pred = np.array([2.5, 5, 4, 8])
        mse_val = root_log_mse(y_pred, y_true)
        self.assertEqual(mse_val, 0.1993)
        
    
    def test_mean_absolute_error_metric(self):
        """
        Test the mean absolute error metric
        """
        mae = MeanAbsoluteError()
        mae_val = mae(np.array([1, 0, 1]), np.array([1, 1, 1]))
        self.assertEqual(mae_val, 0.3333)
        
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mae_val = mae(y_pred, y_true)
        self.assertEqual(mae_val, 0.5)
        
    def test_r2_metric(self):
        """
        Test the r2 metric
        """
        r2 = R2()
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        r2_val = r2(y_pred, y_true)
        self.assertEqual(r2_val, 0.9486)
        
    def test_auc_metric(self):
        """
        Test the auc metric
        """
        test_case = roc_auc_score(self.cancer_y, self.clf.predict_proba(self.cancer_x)[:, 1])
        
        auc = Auc()
        y_true = self.cancer_y
        y_pred = self.clf.predict_proba(self.cancer_x)[:, 1]
        auc_val = auc(y_pred, y_true)
        self.assertEqual(auc_val, round(test_case, 4))
        
    def test_sequential_using_iris_dataset(self):
        """
        Test the sequential model using iris dataset
        """
        train_data = TensorDataset(torch.tensor(self.ir_x), torch.tensor(self.ir_y))

        train_dataloader = DataLoader(train_data, batch_size=32, num_workers=4)
        
        self.iris_model = Sequential(
            [
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.LogSoftmax(dim=1),
            ]
        )

        self.iris_model.compile(
            optimizer="adam",
            loss="nll",
            metrics=[Auc()],
        )

        history = self.iris_model.fit(
            train_dataloader, 
            epochs=1,
            batch_size=1,
            validation_split = 0.2
            )
        
    