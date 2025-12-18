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
logger = setup_logging('categorical_to_numeric')


def cat_to_num_train(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train-time categorical to numerical conversion.
    Learns:
    - Brand target encoding from Car_Name
    """
    logger.info("Starting categorical -> numeric ")
    logger.info(f'{X_train.isnull().sum()}')
    X_train = X_train.copy()

    # ---------------- CAR_NAME → BRAND ----------------
    X_train["Brand"] = X_train["Car_Name"].str.split().str[0].str.lower()
    brand_price_map = y_train.groupby(X_train["Brand"]).mean()
    global_mean = y_train.mean()

    X_train["Brand_mean"] = X_train["Brand"].map(brand_price_map)
    X_train["Brand_mean"].fillna(global_mean, inplace=True)

    # Drop original columns
    X_train.drop(columns=["Car_Name", "Brand"], inplace=True)


    with open("brand_price_map.pkl", "wb") as f:
        pickle.dump(brand_price_map, f)

    with open("global_mean.pkl", "wb") as f:
        pickle.dump(global_mean, f)

    # ---------------- LABEL ENCODING ----------------
    # Fill any missing categorical values first


    X_train["Fuel_Type"] = X_train["Fuel_Type"].map({"Petrol": 1, "Diesel": 0, "CNG":2})
    X_train["Seller_Type"] = X_train["Seller_Type"].map({"Dealer": 1, "Individual": 0})
    X_train["Transmission"] = X_train["Transmission"].map({"Manual": 1, "Automatic": 0})

    logger.info(f"Train columns after encoding: {X_train.columns} -> {X_train.shape}")
    logger.info(f'{X_train.isnull().sum()}')

    logger.info(f"Sample data:{X_train.head(5)}")
    logger.info("Categorical -> numeric (TRAIN) completed")


    return X_train, brand_price_map, global_mean



def cat_to_num_test(X_test: pd.DataFrame, brand_price_map, global_mean):
    """
    Test-time categorical to numerical conversion.
    Uses mappings learned from training.
    """
    logger.info("Starting categorical -> numeric ")
    X_test = X_test.copy()

    # ---------------- CAR_NAME → BRAND ----------------
    X_test["Brand"] = X_test["Car_Name"].str.split().str[0].str.lower()
    X_test["Brand_mean"] = X_test["Brand"].map(brand_price_map)
    X_test["Brand_mean"].fillna(global_mean, inplace=True)

    X_test.drop(columns=["Car_Name", "Brand"], inplace=True)

    # ---------------- LABEL ENCODING ----------------
    # Fill any missing categorical values first


    X_test["Fuel_Type"] = X_test["Fuel_Type"].map({"Petrol": 1, "Diesel": 0 , "CNG":2})
    X_test["Seller_Type"] = X_test["Seller_Type"].map({"Dealer": 1, "Individual": 0})
    X_test["Transmission"] = X_test["Transmission"].map({"Manual": 1, "Automatic": 0})

    logger.info(f"Test columns after encoding: {X_test.columns} -> {X_test.shape}")
    logger.info(f"Sample data:{X_test.head(5)}")
    logger.info("Categorical -> numeric completed")

    return X_test
