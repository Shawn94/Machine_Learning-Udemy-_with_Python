# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:45:11 2019
Categorical data - how to categorize your data

@author: shero
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv') # importing external data file
X = dataset.iloc[:,:-1].values    # extract data we need 
Y = dataset.iloc[:,3].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Label string or other data type as computer read only digits
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #LabelEncoding for countries

onehotencoder = OneHotEncoder(categorical_features=[0]) #After we labeled with numbers country names, it became 0,1,2. Problem: One country can`t be greater than another as 0<1<2
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()              #LabelEncoder for 'Purchased' column(Y/N)
Y = labelencoder_X.fit_transform(Y) 