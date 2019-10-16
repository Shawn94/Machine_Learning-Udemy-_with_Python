# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:43:55 2019
How to handle missing data
@author: shero
"""

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