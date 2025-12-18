import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import pickle
from log_code import setup_logging
logger = setup_logging('feature_scaling')
from sklearn.preprocessing import StandardScaler



def scale_features(X_train, X_test, save_path="scaler.pkl"):
    """
    Scales all features using StandardScaler.
    Fits ONLY on training data.
    """

    try:
        logger.info("Starting feature scaling")

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)

        X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index )

        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("Feature scaling completed and scaler saved")

        return X_train_scaled, X_test_scaled



    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error in Line {error_line.tb_lineno}: {error_msg}")
