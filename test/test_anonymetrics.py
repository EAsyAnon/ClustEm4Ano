import unittest

from anonymetrics.anonymetrics import calculate_k_anonymity, calculate_l_diversity, get_groups, calculate_t_closeness, \
    get_group_sizes, get_count_per_group_size
from anonymize.generalize import *


class TestAnonymetrics(unittest.TestCase):

    def test_calculate_k_anonymity(self):
        # Create a test DataFrame with quasi-identifiers
        df = pd.DataFrame({'age': [30, 30, 40, 40, 50, 50],
                           'gender': ['M', 'M', 'F', 'F', 'M', 'M'],
                           'zipcode': [10001, 10001, 10002, 10002, 10003, 10003],
                           'salary': [50000, 60000, 70000, 80000, 90000, 100000],
                           'siblings': [1, 1, 1, 3, 3, 3]}
                          )

        # Test k-anonymity with a minimum of 1 records per group
        k = calculate_k_anonymity(df, [0, 1, 2, 3])
        self.assertEqual(k, 1)

        # Test k-anonymity with a minimum of 2 records per group
        k = calculate_k_anonymity(df, [0, 1, 2])
        self.assertEqual(k, 2)

        # Test k-anonymity with a minimum of 3 records per group
        k = calculate_k_anonymity(df, [4])
        self.assertEqual(k, 3)

        # Test k-anonymity with a minimum of 1 records per group
        k = calculate_k_anonymity(df, [0, 1, 2, 3, 4])
        self.assertNotEqual(k, 2)

        generalize_categorical(df, [1], ['M', 'F'])
        k = calculate_k_anonymity(df, [0, 1])
        self.assertEqual(k, 2)

    def test_calculate_l_diversity(self):
        # Create a test DataFrame with quasi-identifiers
        df = pd.DataFrame({'age': [30, 30, 40, 40, 50, 50],
                           'gender': ['M', 'M', 'F', 'F', 'M', 'M'],
                           'zipcode': [10001, 10001, 10002, 10002, 10003, 10003],
                           'salary': [50000, 60000, 70000, 80000, 90000, 100000],
                           'siblings': [1, 1, 1, 3, 3, 3]}
                          )

        l = calculate_l_diversity(df, [0, 1, 2], [4])
        self.assertEqual(l, 1)
        generalize_categorical(df, [1], ['M', 'F'])
        # l = calculate_l_diversity(df_generalized, [4], [1])
        # self.assertEqual(l, 1)
        l = calculate_l_diversity(df, [1], [3])
        self.assertEqual(l, 6)

    def test_get_groups(self):
        df = pd.DataFrame({'age': [30, 30, 40, 40, 50, 50],
                           'gender': ['M', 'M', 'F', 'F', 'M', 'M'],
                           'zipcode': [10001, 10001, 10002, 10002, 10003, 10003],
                           'salary': [50000, 60000, 70000, 80000, 90000, 100000],
                           'siblings': [1, 1, 1, 3, 3, 3],
                           'cars': [1, 1, 1, 1, 1, 1]}
                          )

        groups = get_groups(df, qa_indices=[0, 1, 2, 3, 4, 5])
        self.assertEqual(len(groups), 6)

        groups = get_groups(df, qa_indices=[0, 1])
        self.assertEqual(len(groups), 3)

        groups = get_groups(df, qa_indices=[5])
        self.assertEqual(len(groups), 1)

        generalize_categorical(df, [1], ['M', 'F'])
        groups = get_groups(df, qa_indices=[0, 1])
        self.assertEqual(len(groups), 3)

    # see https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf, p.7
    def test_calculate_t_closeness(self):
        data = {'ZIP Code': ['4767*', '4767*', '4767*', '4790*', '4790*', '4790*', '4760*', '4760*', '4760*'],
                'Age': ['<= 40', '<= 40', '<= 40', '>= 40', '>= 40', '>= 40', '>= 40', '>= 40', '>= 40'],
                'Salary': [3000, 5000, 9000, 6000, 11000, 8000, 4000, 7000, 10000],
                'Disease': ['gastric ulcer', 'stomach cancer', 'pneumonia', 'gastritis', 'flu', 'bronchitis',
                            'gastritis', 'bronchitis', 'stomach cancer']}

        df = pd.DataFrame(data=data, index=[1, 3, 8, 4, 5, 6, 2, 7, 9],
                          columns=['ZIP Code', 'Age', 'Salary', 'Disease'])

        # Test with numerical attribute
        t = calculate_t_closeness(df, qa_indices=[0, 1], sa_index=2)
        self.assertAlmostEqual(t, 0.16666666666666669)

        # Test with categorical attribute
        t = calculate_t_closeness(df, qa_indices=[0, 1], sa_index=3)
        self.assertAlmostEqual(t, 0.5555555555555556)

        # generalize_categorical(df, [1], ['<= 40', '>= 40'])
        # t = calculate_t_closeness(df, qa_indices=[0, 1], sa_index=1)
        # self.assertAlmostEqual(t, 0.0)

    def test_get_group_size(self):
        df = pd.DataFrame({'age': [30, 30, 40, 40, 50, 50, 20],
                           'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'D'],
                           'zipcode': [10001, 10001, 10002, 10002, 10003, 10003, 10001],
                           'salary': [50000, 60000, 70000, 80000, 90000, 100000, 1],
                           'siblings': [1, 1, 1, 3, 3, 3, 4]})

        qa_indices = [1]  # gender, zipcode

        # Call the get_group_sizes function
        group_sizes = get_group_sizes(df, qa_indices)

        expected = np.array([1., 2., 4.])

        np.testing.assert_array_equal(group_sizes, expected)

        # counts = get_count_per_group_size(df, qa_indices)
        # plot_count_per_group_size(counts)

        discretize(df, 0, 10.0)
        generalize_categorical(df, [1], ['F', 'M'])

        counts = get_count_per_group_size(df, qa_indices)
        # plot_count_per_group_size(counts)

        expected = np.array([0., 1., 0., 0., 0., 0., 6., 0.])

        # Assert that the calculated group sizes match the expected group sizes
        np.testing.assert_array_equal(counts, expected)

    def test_get_count_per_group_size(self):
        df = pd.DataFrame({'age': [30, 30, 40, 40, 50, 50, 20],
                           'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'D'],
                           'zipcode': [10001, 10001, 10002, 10002, 10003, 10003, 10001],
                           'salary': [50000, 60000, 70000, 80000, 90000, 100000, 1],
                           'siblings': [1, 1, 1, 3, 3, 3, 4]})

        qa_indices = [1, 2]  # gender, zipcode

        expected_count_per_group_size = np.array([0, 1, 6, 0])

        count_per_group_size = get_count_per_group_size(df, qa_indices)

        np.testing.assert_array_equal(expected_count_per_group_size, count_per_group_size)


if __name__ == '__main__':
    unittest.main()
