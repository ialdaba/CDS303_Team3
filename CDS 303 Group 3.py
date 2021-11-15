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
df = pd.read_csv (r'C:\Users\Owner\Desktop\loan_applications.csv')   


# create function to manage outliers
# df = dataframe
# col = column in df
# max = if a value is greater than the max, it is an outlier
# min = if a value is less than the min, it is an outlier
# returns new df with no outliers
def outliers(df, col, dmax, dmin, impute_val):

	outlier = (df[col] > dmax) | (df[col] < dmin)

    # replace all outliers
	df.loc[outlier, col] = impute_val
	return df

dataimputer = SimpleImputer(strategy='median')

datafinal = pd.DataFrame(dataimputer.fit_transform(df))
datafinal.columns = df.columns
datafinal.index = df.index
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
    dfinal = pd.DataFrame(pca.components_, columns=list(trainingset.columns))
    weightamn = dfinal.iloc[i].sort_values(ascending=False)
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
    df: created dataframe
    '''
    df_cols = []
    for i in range(princ_count):
        column_name = columnprefix + '_' + str(i)
        df_cols.append(column_name)
    df = pd.DataFrame(princ_data, columns = df_cols)
    
    return df

pca_datafinal = form_dataframe(final_pca, pca_40.components_.shape[0], 'pc')
pca_datafinal['TARGET'] = training_labels
pca_datafinal.shape
pca_datafinal.head()

