import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('model_training')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def linear_regression(X_train, X_test, y_train, y_test):
    try:
        logger.info("Starting Linear Regression Training")
        logger.info(f'{X_train.isnull().sum()}')

        reg_lr = LinearRegression()
        reg_lr.fit(X_train, y_train)
        y_train_pred = reg_lr.predict(X_train)
        y_test_pred = reg_lr.predict(X_test)

        train_data_r2_score = r2_score(y_train, y_train_pred)
        test_data_r2_score = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        logger.info(f"Train R2 Score : {train_data_r2_score}")
        logger.info(f"Test R2 Score  : {test_data_r2_score}")
        logger.info(f"MAE            : {mae}")
        logger.info(f"RMSE           : {rmse}")


        with open('linear_regression_model.pkl', 'wb') as f:
            pickle.dump(reg_lr, f)

        logger.info("Linear Regression model saved successfully")

        return reg_lr

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error in Line {error_line.tb_lineno}: {error_msg}" )

