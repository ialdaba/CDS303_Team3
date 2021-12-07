import numpy as np
import pandas as pd
import gc
from PIL import Image
import itertools
import pprintpp
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

from explainerdashboard import ExplainerDashboard, ClassifierExplainer

# load cleaned data for applicants
datafinal = pd.read_csv (r'/Users/jeremya./Desktop/cleaned_data.csv')

############################################################################################################
# FEATURE SCALING SECTION

datascaler = MinMaxScaler(feature_range = (0, 1))

datafinal_scaled = datascaler.fit_transform(datafinal)
datafinal_scaled = pd.DataFrame(datafinal_scaled, columns=datafinal.columns)

############################################################################################################
# DATA SPLIT SECTION

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
	X = d.drop([op_col], axis =1)
	# split into training data, testing data 
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3, random_state =35)
	return X_train, X_test, Y_train, Y_test
  
############################################################################################################
# DATA TRAINING SECTION

X_train, X_test, Y_train, Y_test = split(datafinal_scaled, 'TARGET')
model = LogisticRegression()
model.fit(X_train, Y_train)	

############################################################################################################
# DASHBOARD SECTION

explainer = ClassifierExplainer(model, X_test, Y_test, labels=['On Time', 'Not On Time'])
ExplainerDashboard(explainer).run()