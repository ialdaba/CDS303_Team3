import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb

# Code outline for the heat maps was sent to Isabel from Bharath using Discord
# Isabel then inputted the features to be used

col_list = ["EXT_SOURCE_3", "EXT_SOURCE_2", "NAME_INCOME_TYPE_Pensioner", "AMT_CREDIT", "AMT_GOODS_PRICE", "FLAG_EMP_PHONE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_DOCUMENT_3", "ORGANIZATION_TYPE_XNA", "NAME_EDUCATION_TYPE_Secondary___secondary_special", "FLAG_DOCUMENT_3"]
datafinal = pd.read_csv (r'/Users/jeremya./Desktop/data_balanced.csv', usecols=col_list)

dataplot = sb.heatmap(datafinal.corr())

mp.show()

col_list = ["EXT_SOURCE_3", "EXT_SOURCE_2","FLAG_EMP_PHONE", "AMT_GOODS_PRICE", "CODE_GENDER", "FLAG_OWN_CAR"]
datafinal = pd.read_csv (r'/Users/jeremya./Desktop/data_balanced.csv', usecols=col_list)

dataplot = sb.heatmap(datafinal.corr())

mp.show()