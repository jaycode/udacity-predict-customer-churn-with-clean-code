import pytest
import os
import logging
import churn_library as cls
import pdb

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def assert_true(condition, message_when_fail):
    try:
        assert(condition)
    except AssertionError as err:
        logging.error(message_when_fail)
        # err.args requires a set
        err.args = (message_when_fail,)
        raise err


def assert_equal(a, b, message_when_fail="Values not equal"):
    try:
        assert(a == b)
    except AssertionError as err:
        additional_details = f"{a} vs {b}"
        msg = f"{message_when_fail} ({additional_details})"
        logging.error(msg)
        # err.args requires a set
        err.args = (msg,)
        raise err


@pytest.fixture
def df():
    df = cls.import_data("./data/bank_data.csv")
    return df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''

    IMAGE_DIR_EDA = './test_images/eda/'
    cls.perform_eda(df, image_dir_eda=IMAGE_DIR_EDA)
    assert_true(os.path.isfile("./test_images/eda/age_hist.png"),
                "Age history plot does not exist.")


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
                "X_train does not have the correct number of rows.")
    assert_equal(len(X_test), round(len(df) * test_size),
                "X_test does not have the correct number of rows.")
    assert_equal(len(y_train), round(len(df) * (1-test_size)),
                "y_train does not have the correct number of rows.")
    assert_equal(len(y_test), round(len(df) * test_size),
                "y_test does not have the correct number of rows.")


# This decorator makes sure the print outputs are displayed
@pytest.mark.no_capture
def test_train_models(df):
    '''
    test train_models
    '''
    response = 'Churn'
    test_size = 0.4
    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(df, response, test_size=test_size)
    IMAGE_DIR_RESULTS = 'test_images/results/'
    MODEL_DIR = 'test_models/'
    cls.train_models(X_train, X_test, y_train, y_test,
                     param_grid_rfc={'n_estimators': [200, 500],
                                     'max_features': ['auto'],
                                     'max_depth' : [4],
                                     'criterion' :['gini']},
                     image_dir_results=IMAGE_DIR_RESULTS,
                     model_dir=MODEL_DIR)
