# Great is the holy of holies, the LORD of host, the whole earth is full of his glory.

# Importing necessary modules
from contexts import *
import time
import unittest as ut
from exttorch.utils import ProgressBar

class TestProgressBar(ut.TestCase):
    
    def test_progress_bar(self):
        """
        Test the progress bar
        """
        pb = ProgressBar(bar_width=10, show_val_metrics=False)
        for i in range(1000):
            pb.total = 1000
            pb.update(i + 1, metric={"loss": i, "accuracy": i}.items())
            time.sleep(0.2)