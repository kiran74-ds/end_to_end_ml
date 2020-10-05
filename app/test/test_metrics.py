# pylint: disable=missing-module-docstring
import unittest
from sklearn.metrics import precision_score, recall_score
from app.src.train import prepare_data_and_train_model


class TestModelMetrics(unittest.TestCase):
    ''' Test case to check performance of ml model'''
    def test_model_precision_score_should_be_above_threshold(self):
        ''' Test case to check performance of ml model based on Precision '''
        # pylint: disable=invalid-name
        model, _, X_test, Y_test = prepare_data_and_train_model()
        Y_pred = model.predict(X_test)
        precision = precision_score(Y_test, Y_pred)

        self.assertGreaterEqual(precision, 0.6)

    def test_model_recall_score_should_be_above_threshold(self):
        ''' Test case to check performance of ml model based on Precision '''
        # pylint: disable=invalid-name
        model,_, X_test, Y_test = prepare_data_and_train_model()
        Y_pred = model.predict(X_test)
        recall = recall_score(Y_test, Y_pred)

        self.assertGreaterEqual(recall, 0.6)
