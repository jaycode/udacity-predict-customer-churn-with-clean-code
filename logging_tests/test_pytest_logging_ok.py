import pytest
import logging
import pandas as pd

# This method does not work
# def pytest_configure(config):
#     logging.basicConfig(
#         filename='./churn_library_pytest_configure.log',
#         level=logging.INFO,
#         filemode='w',
#         format='%(name)s - %(levelname)s - %(message)s'
#     )

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
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


# By setting the ini file (pytest.ini), simply running the following command would create the log:
# pytest test_pytest_logging_ok.py
#
# Alternative 1:
# With the following code, you should run pytest by running this script with the `python` command.
#
# Alternative 2:
# you could also run the following command:
# pytest test_pytest_logging_ok.py --log-level=INFO --log-file=./churn_library_pytest_command.log

def main():
    options = [
        '--log-level=INFO',
        '--log-file=./churn_library_pytest_main.log',
        __file__
    ]
    pytest.main(options)

if __name__ == '__main__':
    main()
