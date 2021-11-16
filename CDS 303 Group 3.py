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
		print("List of unique outlier values in '{}' column, as found: {}".format(col_name, str(outliers)))
        
		# replace outliers
		loan_apps.loc[is_outlier, col_name] = val
		print('Number of rows updated: ', str(num_of_outlier_rows))
	else:
		print("No outlier found for column '{}'.".format(col_name))
    
	del loan_apps_anomalous
	gc.collect()
    
	return loan_apps
    
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
min_rating = 0 # take some arbitrary high value
loan_apps = handle_outliers(loan_apps, 'REGION_RATING_CLIENT', max_rating, min_rating)

max_income = 1e8
min_income = 0
loan_apps = handle_outliers(loan_apps, 'AMT_INCOME_TOTAL', max_income, min_income)

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

loan_apps = pd.get_dummies(loan_apps)

############################################################################################################
# IMPUTING SECTION

dataimputer = SimpleImputer(strategy='median')

datafinal = pd.DataFrame(dataimputer.fit_transform(loan_apps))
datafinal.columns = loan_apps.columns
datafinal.index = loan_apps.index
datafinal.head()
datafinal.shape

print('Is there any NaN: ', np.any(np.isnan(datafinal)))

datascaler = MinMaxScaler(feature_range = (0, 1))

datafinal_scaled = datascaler.fit_transform(datafinal)
datafinal_scaled = pd.DataFrame(datafinal_scaled, columns=datafinal.columns)
datafinal_scaled.head()

# keep training labels separate
training_labels = datafinal_scaled['TARGET']

# drop training labels from training set
datafinal_scaled_no_tlabel = datafinal_scaled
datafinal_scaled_no_tlabel = datafinal_scaled_no_tlabel.drop(['TARGET'], axis=1)
datafinal_scaled_no_tlabel.shape

pca = PCA()
pca.fit(datafinal_scaled_no_tlabel)


def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    '''
    num_components=len(pca.explained_variance_ratio_)
    index = np.arange(num_components)
    value = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(value)
    ax.bar(index, value)
    ax.plot(index, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(value[i]*100)[:4])), (index[i]+0.2, value[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
pca_40 = PCA(n_components=40)
final_pca = pca_40.fit_transform(datafinal_scaled_no_tlabel)
pca_40.explained_variance_ratio_.sum()

print('Reduced dimensionality of the data: ', final_pca.shape)

def pca_weights(pca, i, trainingset):
    '''
    Map weights for the ith principal component to corresponding feature names,
    and return the sorted list of weights.
    INPUT:
    pca: the PCA component
    i: location of the component
    trainingset: training dataset
    OUTPUT:
    weighted: weights associated with dataset
    '''
    loan_apps_final = pd.DataFrame(pca.components_, columns=list(trainingset.columns))
    weightamn = loan_apps_final.iloc[i].sort_values(ascending=False)
    return weightamn

def plot_PCA_feature_associations(pca, i, trainingset):
    '''
    Map weights for the ith principal component to corresponding feature names,
    and then plot linked values, sorted by weight.
    INPUT:
    pca: the PCA component
    i: location of the component
    trainingset: training dataset
    output: plot of the weights for each prinicpal component
    '''
    row = pca_weights(pca, i, trainingset)
    row.plot(kind='bar', figsize=(18, 8))
    plt.show()
    
plot_PCA_feature_associations(pca_40, 0, datafinal_scaled_no_tlabel)
plot_PCA_feature_associations(pca_40, 1, datafinal_scaled_no_tlabel)
plot_PCA_feature_associations(pca_40, 2, datafinal_scaled_no_tlabel)
    
def form_dataframe(princ_data, princ_count, columnprefix):
    '''
    Form and transform data from n-dimensional data.
    Column names will be formed from given prefix and internal counter value
    INPUT:
    princ_data: n-dimensional array of data (output of PCA)
    princ_count: count of principal components
    columnprefix: prefix for the column names
    OUTPUT:
    apps_df: final dataframe
    '''
    df_cols = []
    for i in range(princ_count):
        column_name = columnprefix + '_' + str(i)
        df_cols.append(column_name)
    df = pd.DataFrame(princ_data, columns = df_cols)
    
    return apps_df

pca_datafinal = form_dataframe(final_pca, pca_40.components_.shape[0], 'pc')
pca_datafinal['TARGET'] = training_labels
pca_datafinal.shape
pca_datafinal.head()

