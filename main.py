'''
In this Project we are finding the calories burnt prediction using regression models
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from handling_missing_values import RSI_tecnique
from variable_transformation import VARIABLE_TRANSFORMATION
from sklearn.preprocessing import OneHotEncoder
import pickle
from categorical_to_numeric import cat_to_num_train, cat_to_num_test
from feature_scaling import scale_features
from model_training import linear_regression






class CAR_PRICE_PREDICTION:
    try:
        def __init__(self,path):
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'{self.df.columns} -> {self.df.shape}')
            self.X = self.df.drop('Selling_Price', axis=1)
            self.y = self.df['Selling_Price']
            #logger.info(f'Columns and datatypes:{self.df.info()}')
            logger.info(f'X - Shape and columns: {self.X.shape} -> {self.X.columns}')
            logger.info(f'y - Shape and columns: {self.y.shape} ')

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
            logger.info(f'{self.X_train.columns}')
            #logger.info(self.y_train.info())

            logger.info(f'{self.X_train.head(5)}')
            logger.info(f'{self.y_train.head(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')
            logger.info(f'Total no.of null values in data:{self.df.isnull().sum()}')

            logger.info(f'========================================')

        def missing_values(self):
            try:
                if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                    self.X_train, self.X_test = RSI_tecnique.random_sample_imputataion(self.X_train, self.X_test)
                else:
                    logger.info(f'There are no null values in data:{self.X_train.isnull().sum()}')
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def var_tranformation(self):
            try:
                logger.info(f'columns of X_train:{self.X_train.columns}')
                logger.info(f'columns of X_test:{self.X_test.columns}')
                self.X_train_numeric = self.X_train.select_dtypes(exclude='object')
                self.X_train_categorical = self.X_train.select_dtypes(include='object')
                self.X_test_numeric = self.X_test.select_dtypes(exclude='object')
                self.X_test_categorical = self.X_test.select_dtypes(include='object')
                logger.info(f'Numerical Columns : {self.X_train_numeric.columns} -> {self.X_train_numeric.shape}')
                logger.info(f'Categorical columns : {self.X_train_categorical.columns}-> {self.X_train_categorical.shape}')
                logger.info(f'Numerical Columns : {self.X_test_numeric.columns} -> {self.X_test_numeric.shape}')
                logger.info(f'Categorical columns : {self.X_test_categorical.columns} -> {self.X_test_categorical.shape}')
                self.X_train_numeric, self.X_test_numeric = VARIABLE_TRANSFORMATION.variable_trans(self.X_train_numeric, self.X_test_numeric)
                logger.info(f"{self.X_train_numeric.columns} -> {self.X_train_numeric.shape}")
                logger.info(f"{self.X_test_numeric.columns} -> {self.X_test_numeric.shape}")
                logger.info(f'{self.X_train.isnull().sum()}')
                logger.info(f'{self.X_test.isnull().sum()}')
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def categorical_to_numerical(self):
            try:
                logger.info("Starting categorical -> numeric conversion")

                # Encode TRAIN
                self.X_train, self.brand_map, self.global_mean = cat_to_num_train(self.X_train, self.y_train)

                self.X_test = cat_to_num_test(self.X_test, self.brand_map, self.global_mean)

                logger.info("Brand map and global mean saved")

                logger.info(f'After encoding, X_train shape: {self.X_train.shape}')
                logger.info(f'After encoding, X_test shape: {self.X_test.shape}')

                # Check types to confirm everything is numeric
                logger.info(f'X_train dtypes:{self.X_train.dtypes}')
                logger.info(f'{self.X_train.isnull().sum()}')
                logger.info(f'{self.X_test.isnull().sum()}')

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
        '''
        def impute_missing_values(self):
            """
            Impute missing values for numeric and categorical parts separately.
            Numeric columns → median
            Categorical/encoded columns → constant 0
            """
            try:
                logger.info("Starting missing value imputation")
                # Impute numeric columns
                numeric_imputer = SimpleImputer(strategy='median')
                self.X_train_numeric = pd.DataFrame(numeric_imputer.fit_transform(self.X_train_numeric),columns=self.X_train_numeric.columns,index=self.X_train_numeric.index)
                self.X_test_numeric = pd.DataFrame(numeric_imputer.transform(self.X_test_numeric),columns=self.X_test_numeric.columns,index=self.X_test_numeric.index)

                # Impute categorical (encoded) columns
                categorical_imputer = SimpleImputer(strategy='constant', fill_value=0)
                self.X_train_categorical = pd.DataFrame(categorical_imputer.fit_transform(self.X_train_categorical),columns=self.X_train_categorical.columns,index=self.X_train_categorical.index)
                self.X_test_categorical = pd.DataFrame(categorical_imputer.transform(self.X_test_categorical),columns=self.X_test_categorical.columns,index=self.X_test_categorical.index)

                # Save the imputers
                with open('imputer_numeric.pkl', 'wb') as f:
                    pickle.dump(numeric_imputer, f)
                with open('imputer_categorical.pkl', 'wb') as f:
                    pickle.dump(categorical_imputer, f)


                # Verify no missing values remain
                logger.info(
                    f"Missing values in X_train_numeric after imputation: {self.X_train_numeric.isnull().sum().sum()}")
                logger.info(
                    f"Missing values in X_test_numeric after imputation: {self.X_test_numeric.isnull().sum().sum()}")
                logger.info(
                    f"Missing values in X_train_categorical after imputation: {self.X_train_categorical.isnull().sum().sum()}")
                logger.info(
                    f"Missing values in X_test_categorical after imputation: {self.X_test_categorical.isnull().sum().sum()}")

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.error(f"Error in Line {error_line.tb_lineno}: {error_msg}")
            '''
        def feature_s(self):
            try:
                logger.info('Starting feature scaling')
                self.X_train, self.X_test = scale_features(self.X_train, self.X_test, save_path="scaler.pkl")
                logger.info(self.X_train.head(4))
                logger.info(self.X_test.head(4))

                logger.info('Scaling Data After Regression')

                logger.info(f'=========================== X - Training Data==============================================')
                logger.info(f'X_train columns : {self.X_train.columns}')
                logger.info(f'X_test columns: {self.X_test.columns}')
                logger.info(f'X_train : {self.X_train.shape}')
                logger.info(f'X_test : {self.X_test.shape}')
                logger.info(self.X_train.head(4))
                logger.info(self.X_test.head(4))
                logger.info(f'{self.X_train.isnull().sum()}')
                logger.info(f'{self.X_test.isnull().sum()}')



            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')



        def model_training(self):
            try:
                logger.info('Training Model')
                linear_regression(self.X_train, self.X_test, self.y_train, self.y_test)

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')



    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')















if __name__ == '__main__':
    try:
            obj = CAR_PRICE_PREDICTION('C:\\Users\\VARSHINI\\Downloads\\Car_price_prediction\\car_price.csv')
            obj.missing_values()
            obj.var_tranformation()
            obj.categorical_to_numerical()
            obj.feature_s()
            obj.model_training()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')