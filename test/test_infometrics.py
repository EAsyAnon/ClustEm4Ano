import unittest
import pandas as pd

from anonymize.generalize import generalize_categorical, discretize
from anonymetrics.infometrics import numerical_info_loss, entropy_info_loss, euclid_info_loss


class TestInfometrics(unittest.TestCase):

    def setUp(self):

        # Define the example data from the Adult datasets
        self.df2 = pd.DataFrame({
            'age': [39, 50, 38, 53, 28, 37],
            'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private', 'Private'],
            'fnlwgt': [77516, 83311, 215646, 234721, 338409, 284582],
            'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors', 'Masters'],
            'education-num': [13, 13, 9, 7, 13, 14],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-civ-spouse',
                               'Married-civ-spouse', 'Married-civ-spouse'],
            'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Handlers-cleaners',
                           'Prof-specialty', 'Exec-managerial'],
            'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife', 'Wife'],
            'race': ['White', 'White', 'White', 'Black', 'Black', 'White'],
            'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female'],
            'capital-gain': [2174, 0, 0, 0, 0, 0],
            'capital-loss': [0, 0, 0, 0, 0, 0],
            'hours-per-week': [40, 13, 40, 40, 40, 40],
            'native-country': ['United-States', 'United-States', 'United-States', 'United-States', 'Cuba',
                               'United-States'],
            'income': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
        })

        self.df1 = self.df2.copy()

        discretize(self.df2, 0, 10.0)
        generalize_categorical(self.df2, [3], ['Bachelors', 'Masters'])
        # generalize_categorical(self.df2, [3], ['HS-grad', '11th'])

    def test_numerical_info_loss(self):
        infoloss = numerical_info_loss(self.df1, self.df2, 0)
        self.assertAlmostEqual(infoloss, 0.36)
        infoloss = numerical_info_loss(self.df2, self.df2, 0)
        self.assertAlmostEqual(infoloss, 0.00)
        infoloss = numerical_info_loss(self.df1, self.df1, 0)
        self.assertAlmostEqual(infoloss, 0.00)

    def test_entropy_info_loss(self):
        infoloss = entropy_info_loss(self.df1, self.df2, 3)
        self.assertAlmostEqual(infoloss, 3.2451124978365313)

    def test_euclid_info_loss(self):
        infoloss = euclid_info_loss(self.df1, self.df1, 0)
        self.assertEqual(infoloss, 0.0)
        infoloss = euclid_info_loss(self.df1, self.df2, 0)
        self.assertAlmostEqual(infoloss, 3.3333333333333335)


if __name__ == '__main__':
    unittest.main()
