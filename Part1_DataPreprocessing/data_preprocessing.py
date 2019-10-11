# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:42:04 2019

@author: shero
"""

#Data Preprocessing part

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv') # importing external data file
X = dataset.iloc[:,:-1].values    # extract data we need 
Y = dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.impute import SimpleImputer            # sklearn.imputer -> this is to solve problem with missing data
imputer = SimpleImputer(missing_values= np.NaN, strategy='mean') # find missing data and put mean value
X[:,1:3] = imputer.fit_transform(X[:,1:3]) # transform missing data to mean

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Label string or other data type as computer read only digits
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #LabelEncoding for countries

onehotencoder = OneHotEncoder(categorical_features=[0]) #After we labeled with numbers country names, it became 0,1,2. Problem: One country can`t be greater than another as 0<1<2
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()              #LabelEncoder for 'Purchased' column(Y/N)
Y = labelencoder_X.fit_transform(Y)