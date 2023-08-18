import logging
import os
import pytest

LOG_DIR = './logs'
LOG_FILENAME = 'churn_library.log'
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

def test_sample():
    logger.info("Test sample function executed.")
    assert 1 == 1

def test_another_sample():
    logger.info("Another test sample function executed.")
    assert 2 == 2
