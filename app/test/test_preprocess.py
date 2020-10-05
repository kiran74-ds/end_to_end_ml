# pylint: disable=missing-module-docstring

import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from app.src.preprocess import impute_missing_values, train_model

class TestProcessing(unittest.TestCase):
    ''' Test cases for preprocessing techniques'''
    @staticmethod
    def test_impute_nans_for_categorical_columns():
        # pylint: disable=missing-function-docstring
        data_frame = pd.DataFrame({
            'categorical_column': ['A', 'A', 'B', np.nan, 'A', np.nan]
        })

        expected = pd.DataFrame({
            'categorical_column': ['A', 'A', 'B', 'A', 'A', 'A']
        })

        assert_frame_equal(expected, impute_missing_values(
            data_frame, ['categorical_column']))

    @staticmethod
    def test_impute_nans_for_numerical_columns():
        # pylint: disable=missing-function-docstring
        data_frame = pd.DataFrame({'numeric_column': [10, 20, np.nan, np.nan, 30]})
        expected = pd.DataFrame({'numeric_column': [10, 20, 20, 20, 30]})

        assert_frame_equal(expected, impute_missing_values(data_frame, ['numeric_column'],
                                                           True), check_dtype=False)


    def test_train_model(self):
        # pylint: disable=missing-function-docstring
        classifiers = [DecisionTreeClassifier,GradientBoostingClassifier, LogisticRegression]
        for classifier in classifiers:
            model, accuracy = train_model(classifier, [[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                                       [1, 1, 1]], [0, 1, 0, 1])

            self.assertIsInstance(model, classifier)
            self.assertIsInstance(accuracy, float)
