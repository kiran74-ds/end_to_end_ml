# pylint: disable=missing-module-docstring
import logging
import pandas as pd
from sklearn.metrics import classification_report

def impute_missing_values(data_frame, columns, is_numeric=False):
    '''
    Imputing Missing Values, Numeric with median, Categorical with Mode
    args:data_frame : data frame
        columns: list of column names
        is_numeric : Boolean
    return -- dataframe
    '''
    if is_numeric:
        logging.info("Imputing Missing Values for Numerical Columns")
    else:
        logging.info("Imputing Missing Values for Categorical Columns")
    for column in columns:
        if is_numeric:
            data_frame[column] = data_frame[column].fillna(data_frame[column].median())
        else:
            data_frame[column] = data_frame[column].fillna(data_frame[column].mode()[0])

    return data_frame

def normalize_columns(data_frame, columns):
    '''Normalizing columns
    args:
        data_frame : data frame
        columns: list of column names
    return -- dataframe
    '''
    logging.info("Normalizing Numerical Columns")
    for column in columns:
        mean = data_frame[column].mean()
        std = data_frame[column].std()
        if std != 0:
            data_frame[column] = (data_frame[column] - mean) / std
        else:
            data_frame[column] = 0.0
    return data_frame


def one_hot_coding(data_frame, columns):
    '''Creating Dummy Columns
    args:
        data_frame : data frame
        columns: list of column names
    return -- dataframe
        '''
    logging.info("Creating Dummy Varibles")
    for column in columns:
        data_frame = pd.concat([data_frame, pd.get_dummies(data_frame[column],
                                                           prefix=column)], axis=1)
        data_frame.drop(column, inplace=True, axis=1)

    return data_frame

def train_model(model_class, features, target):
    '''
    :param model_class: Machine Learning Classification Algorithm
    :param features: Input features
    :param target: target
    :return: model, model accuracy
    '''

    logging.info("Training Model")

    model = model_class()
    model.fit(features, target)
    predictions = model.predict(features)
    accuracy_score = round(model.score(features, target) * 100, 2)
    print(f'accuracy ({model.__repr__()}): {accuracy_score}')
    logging.info(classification_report(target, predictions))

    return model, accuracy_score

def prepare_data(data_frame):
    '''
    Data Preparation to feed to the machine learning model
    '''
    drop_columns = ['Name','SibSp', 'Parch', 'Ticket', 'Cabin','PassengerId']
    data_frame = data_frame.drop(columns=drop_columns)
    data_frame = impute_missing_values(data_frame, ['Age', 'Fare'], True)
    data_frame = impute_missing_values(data_frame, ['Embarked'], False)
    data_frame = normalize_columns(data_frame, ['Age', 'Fare'])
    data_frame = one_hot_coding(data_frame, ['Sex', 'Embarked','Pclass'])
    return data_frame
