import numpy as np
import pandas as pd
import gc
from PIL import Image
import itertools
import pprintpp
import random


from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


import seaborn as sns
import matplotlib.pyplot as plt

from explainerdashboard import ExplainerDashboard, ClassifierExplainer

# load cleaned data for applicants; change depending on computer
col_list = ["EXT_SOURCE_3", "EXT_SOURCE_2","FLAG_EMP_PHONE", "AMT_GOODS_PRICE", "CODE_GENDER", "FLAG_OWN_CAR", "TARGET"]
datafinal = pd.read_csv (r'/Users/jeremya./Desktop/data_balanced.csv', usecols=col_list)

############################################################################################################
# FEATURE SCALING SECTION  (George)

datascaler = MinMaxScaler(feature_range = (0, 1))

datafinal_scaled = datascaler.fit_transform(datafinal)
datafinal_scaled = pd.DataFrame(datafinal_scaled, columns=datafinal.columns)

############################################################################################################
# DATA SPLIT SECTION (Taylor)

def split(d,op_col):
	'''
	Input
	d: data
	op_col: output column with labels 
	
	Output
	X_train: input parameters for training 
	X_test: input parameters for testing 
	Y_train: output labels for training 
	Y_test: output labels for testing 
	'''
  
	#create X,Y
	Y = d[op_col]
	X = d.drop([op_col], axis = 1)
	# split into training data, testing data 
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3, random_state =40)
	return X_train, X_test, Y_train, Y_test

# Combined with Leah's
X_train, X_test, Y_train, Y_test = split(datafinal_scaled, 'TARGET')
Y = datafinal_scaled['TARGET']
X = datafinal_scaled.drop(['TARGET'], axis = 1)

############################################################################################################
# DATA FITTING SECTION (Leah)
model = LogisticRegression(max_iter = 100, tol = 0.01)
model.fit(X_train, Y_train)

############################################################################################################
# DATA TUNING SECTION (Isabel)

tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200, 250, 300, 350, 1000]

param_grid = dict(tol = tol, max_iter = max_iter)
grid_model = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)
grid_model_result = grid_model.fit(X, Y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best %f using %s" % (grid_model_result.best_score_,grid_model_result.best_params_))

############################################################################################################
# DASHBOARD SECTION (Isabel)

explainer = ClassifierExplainer(model, X_test, Y_test, labels=['On Time', 'Not On Time'])
ExplainerDashboard(explainer).run()
