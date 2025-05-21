# Great is the holy of holies, the LORD of host, the whole earth is full of his glory.

# Importing necessary modules
from contexts import *
import time
import unittest as ut
from exttorch.utils import ProgressBar


class TestProgressBar(ut.TestCase):
    def test_progressbar(self):
        """
        Test the progress bar functionality
        """
        # Initialize the ProgressBar
        progress_bar = ProgressBar(
            bar_width=20,
            verbose="full",
            fill_style="➤",
            empty_style="▷",
            epochs=2
        )

        # Set the total number of iterations
        total_iterations = 100
        progress_bar.total = total_iterations
        
        for epoch in range(2):
            progress_bar.add_epoch = epoch
            
            # Simulate the progress
            for i in range(total_iterations):
                time.sleep(0.01)
                progress_bar.update(i + 1, [("loss", i / 0.3223), ("accuracy", i / 0.3220)])
