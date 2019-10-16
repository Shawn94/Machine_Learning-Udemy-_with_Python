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



#Splitting the dataset into the Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  """




