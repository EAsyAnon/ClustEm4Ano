import unittest
import pandas as pd

from anonymetrics.anonymetrics import get_groups
from anonymetrics.utilitymetrics import avg_size, c_avg, c_dm
from anonymize.generalize import generalize_categorical, discretize
from anonymetrics.anonymetrics import calculate_k_anonymity


class TestUtilitymetrics(unittest.TestCase):

    def setUp(self):

        # Define the example data from the Adult datasets
        self.df = pd.DataFrame({
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

        discretize(self.df, 4, 5.0)
        generalize_categorical(self.df, [3], ['Bachelors', 'Masters'])
        generalize_categorical(self.df, [3], ['HS-grad', '11th'])

        self.groups = get_groups(self.df, [3, 4])

        self.k = calculate_k_anonymity(self.df, [3, 4])
        print('test ....', self.k)

        print(self.df)

    def test_avg_size(self):
        result = avg_size(self.groups)
        self.assertEqual(result, 3)

    def test_c_avg(self):
        result = c_avg(self.df, self.groups, 2)
        print(result)
        self.assertEqual(result, 1.5)

    def test_c_dm(self):
        result = c_dm(self.groups)
        print(result)
        self.assertEqual(result, 20)


if __name__ == '__main__':
    unittest.main()
