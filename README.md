# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is my submission for the Predict Customer Churn project by Udacity

## Files and data description

- `churn_notebook.pynb`: The notebook that contains unrefactored code.
- `churn_library.py`: The refactored script.
- `churn_script_logging_and_testing.py`: Unit testing file. Uses `pytest`

## Running Files

The project was tested with Python 3.6.3. After setting up the environment to use
Python 3.6.3, run the following command to install all required modules:

```
pip install -r requirements_py3.6.txt
```

To test with `pytest`, run this command:

```
pytest churn_script_logging_and_testing.py
```

Logs of the pytest run are stored in the `./logs/` directory.

Note: If you want to display the print outputs, you may run it with a `-s` argument.

To do a test-run with all functions in the library, simply run it like so:

```
python churn_library.py
```

## Code Quality Metrics

### Follows PEP 8 Guidelines

- Running `pylint ./churn_libraries.py` gave me a score of 7.84.
- Running `pylint ./churn_script_logging_and_tests.py` gave me a score of 7.62.


### Runs Successfully

Running the pytest command above locally produced the following output:

![output](successful_pytest_run.png)

Notice that all five tests ran successfully.
