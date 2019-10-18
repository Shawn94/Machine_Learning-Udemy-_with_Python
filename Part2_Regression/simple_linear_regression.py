# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:05:37 2019

@author: shero
"""
# Simple Linear Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

#Splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#Feature Scaling


#Fitting Simple Linear Regression to ther Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # X_train - Training data(independant value); y_train - Target values(dependant variable)

#Predictiong the Test set results
y_predict = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') #Observation point
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # regression line
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Yeasrs of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red') #Observation point
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # regression line
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Yeasrs of experience')
plt.ylabel('Salary')
plt.show()