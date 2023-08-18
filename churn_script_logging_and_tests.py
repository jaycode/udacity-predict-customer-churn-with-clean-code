'''
The script that contains test functions to be run with `pytest`.

Author: Jay Teguh
Creation Date: 08/17/2023
'''

import os
import logging
import pytest
import churn_library as cls

LOG_DIR = './logs'
LOG_FILENAME = 'churn_library_test.log'
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

file_handler = logging.FileHandler(LOG_PATH, mode='w')
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(file_handler)


def assert_true(condition, message_when_succeed, message_when_fail):
    '''
    Performs an assertion and uses logging for error.
    input:
            condition: The condition to test
            message_when_succeed: message to show when assertion succeeds
            message_when_fail: message to show when assertion fails

    output:
             None
    '''
    try:
        assert condition
        logger.info(message_when_succeed)
    except AssertionError as err:
        logger.error(message_when_fail)
        # err.args requires a set
        err.args = (message_when_fail,)
        raise err


def assert_equal(a, b, message_when_succeed, message_when_fail):
    '''
    Performs an equality assertion.
    input:
        a: variable 1
        b: variable 2
        message_when_succeed: message to show when assertion succeeds
        message_when_fail: message to show when assertion fails

    output:
        None
    '''
    try:
        assert a == b
        logger.info(message_when_succeed)
    except AssertionError as err:
        additional_details = f"{a} != {b}"
        msg = f"{message_when_fail} ({additional_details})"
        logger.error(msg)
        # err.args requires a set
        err.args = (msg,)
        raise err

def remove_all_files(directory):
    '''
    Removes all files in a directory
    input:
        directory: Directory path
    output:
        None
    '''
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


@pytest.fixture
def df():
    '''
    The DataFrame object fixture to use in test functions
    '''
    df = cls.import_data("./data/bank_data.csv")
    return df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''

    image_dir_eda = './test_images/eda/'
    remove_all_files(image_dir_eda)
    cls.perform_eda(df, image_dir_eda=image_dir_eda)
    assert_true(os.path.isfile("./test_images/eda/age_hist.png"),
                "Age history plot created.",
                "Age history plot not created.")


def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    category_lst = [
        'Gender',
    ]
    response = 'Churn'
    df = cls.encoder_helper(df, category_lst, response)
    assert_true('Gender_Churn' in df.columns,
                "Gender_Churn is created.",
                "Gender_Churn is not created.")


def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering
    '''
    response = 'Churn'
    test_size = 0.4
    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(df, response, test_size=test_size)
    assert_equal(len(X_train), round(len(df) * (1-test_size)),
                "X_train has the correct number of rows.",
                "X_train does not have the correct number of rows.")
    assert_equal(len(X_test), round(len(df) * test_size),
                "X_test has the correct number of rows.",
                "X_test does not have the correct number of rows.")
    assert_equal(len(y_train), round(len(df) * (1-test_size)),
                "y_train has the correct number of rows.",
                "y_train does not have the correct number of rows.")
    assert_equal(len(y_test), round(len(df) * test_size),
                "y_test has the correct number of rows.",
                "y_test does not have the correct number of rows.")


def test_train_models(df):
    '''
    test train_models
    '''
    response = 'Churn'
    test_size = 0.4
    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(df, response, test_size=test_size)
    image_dir_results = 'test_images/results/'
    remove_all_files(image_dir_results)

    model_dir = 'test_models/'
    remove_all_files(model_dir)

    cls.train_models(X_train, X_test, y_train, y_test,
                     param_grid_rfc={'n_estimators': [200, 500],
                                     'max_features': ['auto'],
                                     'max_depth' : [4],
                                     'criterion' :['gini']},
                     image_dir_results=image_dir_results,
                     model_dir=model_dir)

    assert_true(os.path.isfile("./test_images/results/classification_results.png"),
                "classification_results.png is created.",
                "classification_results.png is not created.")
    assert_true(os.path.isfile("./test_models/rfc_model.pkl"),
                "rfc_model.pkl is created.",
                "rfc_model.pkl is not created.")
