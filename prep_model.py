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

# load data for applicants
loan_apps = pd.read_csv (r'/Users/jeremya./Desktop/loan_apps.csv')   


# create function to manage outliers
# loan_apps = dataframe
# col = column in loan_apps
# max = if a value is greater than the max, it is an outlier
# min = if a value is less than the min, it is an outlier
# val = value we will temporarily replace with NaN (will change to median later)
# returns new loan_apps with no outliers
def handle_outliers(loan_apps, col_name, max, min, val = np.nan):
	
	is_outlier = (loan_apps[col_name] > max) | (loan_apps[col_name  ] < min)
	loan_apps_anomalous = loan_apps[is_outlier]
	
	num_of_outlier_rows = len(loan_apps_anomalous.index)
    
	if num_of_outlier_rows > 0:
		# print unique outlier values
		outliers = loan_apps_anomalous[col_name].unique()
		print("List of unique outlier values in '{}' column: {}".format(col_name, str(outliers)))
        
		# replace outliers
		loan_apps.loc[is_outlier, col_name] = val
		print('Number of rows updated: ', str(num_of_outlier_rows))
	else:
		print("Zero outliers found in column '{}'.".format(col_name))
    
	del loan_apps_anomalous
	gc.collect()
    
	return loan_apps

# removes rows
def data_cut(df, col_name, values):
   
    before = len(df.index)
    df = df[df[col_name].isin(values)]
    after = len(df.index)
    
    new = before - after
    if new > 0:
        print('Number of rows excluded: ', str(new))
    else:
        print("Zero outliers found in column '{}'.".format(col_name))
    
    return df
    
min_days_birth = -32850
max_days_birth = 0

loan_apps = handle_outliers(loan_apps, 'DAYS_BIRTH', max_days_birth, min_days_birth)

min_employ = -21900
max_employ = 0

loan_apps = handle_outliers(loan_apps, 'DAYS_EMPLOYED', max_employ, min_employ)

max_car_age = 80
min_car_age = 0
loan_apps = handle_outliers(loan_apps, 'OWN_CAR_AGE', max_car_age, min_car_age)

max_rating = 999 # take some arbitrary high value
min_rating = 0 
loan_apps = handle_outliers(loan_apps, 'REGION_RATING_CLIENT', max_rating, min_rating)

max_income = 1e8
min_income = 0
loan_apps = handle_outliers(loan_apps, 'AMT_INCOME_TOTAL', max_income, min_income)

loan_apps['CODE_GENDER'].unique()
gender = ['M','F']
loan_apps = data_cut(loan_apps, 'CODE_GENDER', gender)

############################################################################################################
# ENCODING SECTION

# Create a label encoder object
le = LabelEncoder()
encoded_col_count = 0

# Iterate through the columns
for col in loan_apps:
    if loan_apps[col].dtype == 'object':
        # If 2 unique categories
        if len(list(loan_apps[col].unique())) == 2:
            # Fit and transform column data
            loan_apps[col] = le.fit_transform(loan_apps[col])
            # Keep track of how many columns were label encoded
            encoded_col_count += 1

# One-hot encoding
loan_apps = pd.get_dummies(loan_apps)

############################################################################################################
# IMPUTING SECTION

dataimputer = SimpleImputer(strategy='median')

datafinal = pd.DataFrame(dataimputer.fit_transform(loan_apps))
datafinal.columns = loan_apps.columns
datafinal.index = loan_apps.index
datafinal.head()
datafinal.shape


print('NaN Exists? ', np.any(np.isnan(datafinal)))

datafinal.to_csv("cleaned_data.csv")

############################################################################################################
# FEATURE SCALING SECTION

datascaler = MinMaxScaler(feature_range = (0, 1))

datafinal_scaled = datascaler.fit_transform(datafinal)
datafinal_scaled = pd.DataFrame(datafinal_scaled, columns=datafinal.columns)
datafinal_scaled.head()

############################################################################################################
# DIMENSION REDUCTION SECTION

# keep training labels separate
training_labels = datafinal_scaled['TARGET']

# drop training labels from training set
datafinal_scaled_no_tlabel = datafinal_scaled
datafinal_scaled_no_tlabel = datafinal_scaled_no_tlabel.drop(['TARGET'], axis=1)
datafinal_scaled_no_tlabel.shape

pca = PCA(n_components= 30)
final_pca = pca.fit_transform(datafinal_scaled_no_tlabel)

print ("Proportion of Variance Explained : ", pca.explained_variance_ratio_)  
    
out_sum = pca.explained_variance_ratio_.sum() 
print ("Cumulative Prop. Variance Explained: ", out_sum)

def form_dataframe(princ_data, princ_count, columnprefix):
    '''
    Form and transform data from n-dimensional data.
    Column names will be formed from given prefix and internal counter value
    INPUT:
    princ_data: n-dimensional array of data (output of PCA)
    princ_count: count of principal components
    columnprefix: prefix for the column names
    OUTPUT:
	apps_df: created dataframe
    '''
    loan_apps_cols = []
    for i in range(princ_count):
        column_name = columnprefix + '_' + str(i)
        loan_apps_cols.append(column_name)
    loan_apps = pd.DataFrame(princ_data, columns = loan_apps_cols)
    
    return loan_apps

pca_datafinal = form_dataframe(final_pca, pca.components_.shape[0], 'pc')
pca_datafinal['TARGET'] = training_labels

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

X_train, X_test, Y_train, Y_test = split(pca_datafinal, 'TARGET')
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)	





