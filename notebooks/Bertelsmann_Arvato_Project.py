import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Model selection and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

# Classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

# Additional libraries
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime, date
from time import time,ctime
import itertools
from itertools import product
from scipy.stats import skew
import missingno as msno
from scipy import stats
from subprocess import call
import joblib
import streamlit as st

train_features = joblib.load("model/train_features.pkl")
df_train_feature_names = train_features

def clean_dataframes(df):
    """
    Cleans and preprocesses a given DataFrame by handling missing values, encoding categorical variables,
    and ensuring compatibility with training data when `test=1`.

    Steps involved:
    1. Identifies numeric columns and calculates skewness.
    2. Imputes missing values in skewed numeric columns using the median.
    3. Imputes missing values in normally distributed numeric columns using the mean.
    4. Converts specific categorical columns to string format to prevent inconsistencies.
    5. Extracts the year from the 'EINGEFUEGT_AM' column and converts it to a string.
    6. Applies one-hot encoding to categorical variables.
    7. Ensures compatibility with training features by removing unseen columns when `test=1`.
    8. Removes columns corresponding to missing values ('_nan') to avoid the dummy variable trap.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.
    test (int, optional): If set to 1, ensures the test dataset has the same features as the training dataset. Defaults to 0.

    Returns:
    pd.DataFrame: A cleaned and preprocessed DataFrame ready for modeling.
    """
    # Identify skewed and normal numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    skewness = df[numeric_cols].apply(skew, nan_policy='omit')
    numeric_skewed_cols = skewness[skewness.abs() > 0.5].index  # Adjust threshold as needed
    numeric_normal_cols = skewness[skewness.abs() <= 0.5].index
    
    # Impute skewed columns with median
    skew_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df[numeric_skewed_cols] = skew_imputer.fit_transform(df[numeric_skewed_cols])

    # Impute normal columns with mean
    normal_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[numeric_normal_cols] = normal_imputer.fit_transform(df[numeric_normal_cols])
    
    # Convert categorical columns to strings
    categorical_columns = ["CAMEO_DEU_2015", "CAMEO_DEUG_2015", "CAMEO_INTL_2015", "D19_LETZTER_KAUF_BRANCHE", "EINGEFUEGT_AM", "OST_WEST_KZ"]
    
    for col in categorical_columns:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
        df[col] = df[col].replace("nan", np.nan)  # Convert string "nan" to np.nan
    
    # Impute categorical columns with most_frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    return df

def preprocess_dataframe(df, is_train=False, is_test=False, df_train_feature_names=None):
    """
    Preprocesses a given DataFrame by applying One-Hot Encoding (OHE) to categorical columns
    and ensuring consistency between training and test data.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    is_train (bool, optional): If True, fit and save the ColumnTransformer. Defaults to False.
    is_test (bool, optional): If True, load saved ColumnTransformer and apply transform. Defaults to False.
    df_train_feature_names (list, optional): Feature names from the training set, required for test mode.
    
    Returns:
    pd.DataFrame: A transformed DataFrame with One-Hot Encoding applied and unnecessary columns removed.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    
    if is_train:
        # Fit transformer and save it
        cat_cols = df.select_dtypes(include=['object']).columns
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ct = ColumnTransformer(transformers=[("one_hot_encoder", ohe, cat_cols)], remainder='passthrough')
        
        df_new = ct.fit_transform(df)
        joblib.dump(ct, "model/trained_ct.pkl")
        df_feature_names = ct.get_feature_names_out()

    elif is_test:
        # Load transformer and transform
        ct = joblib.load("model/trained_ct.pkl")
        df_new = ct.transform(df)
        df_feature_names = ct.get_feature_names_out()

    else:
        # For general (non-training/non-test) use
        cat_cols = df.select_dtypes(include=['object']).columns
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ct = ColumnTransformer(transformers=[("one_hot_encoder", ohe, cat_cols)], remainder='passthrough')
        
        df_new = ct.fit_transform(df)
        df_feature_names = ct.get_feature_names_out()

    # Convert to DataFrame
    df_new = pd.DataFrame(df_new, columns=df_feature_names)

    # For test: ensure feature alignment
    if is_test and df_train_feature_names is not None:
        missing_cols = [col for col in df_feature_names if col not in df_train_feature_names]
        df_new = df_new.drop(columns=missing_cols, errors='ignore')

    # Remove dummy trap columns
    df_new = df_new.loc[:, ~df_new.columns.str.endswith("_nan")]

    return df_new