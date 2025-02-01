# Praise Ye The Lord.

# Import libraries
import unittest as ut
from exttorch.hyperparameter import HyperParameters


class TestHyperParameter(ut.TestCase):
    def setUp(self):
        """
        Initialize the hyperparameter
        """
        self.hp = HyperParameters()

    def test_choice_parameter(self):
        """
        Test the choice parameter
        """
        # Add feature choice
        self.hp.Choice("features", [128, 256, 512, 1062])

        # Test if the features is equal to 128
        self.assertEqual(self.hp.features.default, 128)

    def test_int_parameter(self):
        """
        T
        """

    def test_float_parameter(self):
        pass

    def test_boolean_parameter(self):
        pass


if __name__ == "__main__":
    ut.main()
