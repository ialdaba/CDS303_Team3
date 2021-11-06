import numpy as np
import pandas as pd
import gc
from PIL import Image
import itertools
import pprint
import random


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# load in file
df = pd.read_csv (r'/Users/jeremya./Desktop/loan_apps.csv')   


# create function to manage outliers
# df = dataframe
# col = column in df
# max = if a value is greater than the max, it is an outlier
# min = if a value is less than the min, it is an outlier
# returns new df with no outliers
def outliers(df, col, max, min, impute_val):
	
	outlier = (df[col] > max) | (df[col] < min)

    # replace all outliers
	df.loc[outlier, col] = impute_val
    
	return df

