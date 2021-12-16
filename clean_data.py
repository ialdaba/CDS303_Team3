import numpy as np
import pandas as pd
import gc
from PIL import Image
import itertools
import pprintpp
import random


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV


import seaborn as sns
import matplotlib.pyplot as plt

from explainerdashboard import ExplainerDashboard, RegressionExplainer

############################################################################################################
# HANDLING OUTLIERS (Isabel)

# load data for applicants
datafinal = pd.read_csv (r'/Users/jeremya./Desktop/data_balanced.csv', usecols=col_list)  


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
# ENCODING SECTION (George)

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
# IMPUTING SECTION (George)

dataimputer = SimpleImputer(strategy='median')

datafinal = pd.DataFrame(dataimputer.fit_transform(loan_apps))
datafinal.columns = loan_apps.columns
datafinal.index = loan_apps.index
datafinal.head()
datafinal.shape


print('NaN Exists? ', np.any(np.isnan(datafinal)))

datafinal.to_csv("cleaned_data.csv")




