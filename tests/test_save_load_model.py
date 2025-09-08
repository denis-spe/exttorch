# Blessed is the LORD GOD of hosts.

# ============ Test model save and loading of model ============
import unittest
import torch.nn as nn
import os, numpy as np
from sklearn.datasets import make_classification
from src.exttorch.models import StackedModel, load_model_or_weight


class TestModelSaveAndLoad(unittest.TestCase):
    def setUp(self):
        self.x, self.y = make_classification()

        model = StackedModel()
        model.add(nn.Linear(20, 64))
        model.add(nn.Linear(64, 64))
        model.add(nn.Linear(64, 1))
        model.add(nn.Sigmoid())

        model.compile(optimizer="Adam", loss="BCELoss")

        model.fit(self.x, self.y, progress_bar_width=10)

        self.model = model

    def test_model_save(self):
        self.assertTrue(hasattr(self.model, "save"))
        self.model.save("tests/models/model.ext")
        self.assertTrue(os.path.exists("tests/models/model.ext"))

    def test_load_model(self):
        loaded_model = load_model_or_weight("tests/models/model.ext")
        self.assertIsInstance(loaded_model, StackedModel)
        self.assertTrue(hasattr(loaded_model, "predict"))
        pred = loaded_model.predict(self.x)
        self.assertIsInstance(pred, np.ndarray)

    def test_model_weight_save(self):
        self.assertTrue(hasattr(self.model, "save"))
        self.model.save("tests/weights/model.we")
        self.assertTrue(os.path.exists("tests/weights/model.we"))

    def test_load_model_weight(self):
        loaded_weight = load_model_or_weight("tests/weights/model.we")
        self.assertIsInstance(loaded_weight, dict)

        # self.model = Sequential()
        self.model.load_model_state_dict(loaded_weight)
        self.assertIsInstance(self.model, StackedModel)
        pred = self.model.predict(self.x)
        self.assertIsInstance(pred, np.ndarray)
