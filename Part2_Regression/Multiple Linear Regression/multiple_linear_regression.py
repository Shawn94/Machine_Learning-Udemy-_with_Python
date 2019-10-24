# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap D2 = 1 - D1
X = X[:,1:]

#Splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X [:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step2: Fill the full model with all possible predictors(independent variables)
regressor_OLS.summary() # Step 5 Fit model without that variable*

X_opt = X [:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step3-4: Considering highest p-values. if P-value>SL -> remove it
regressor_OLS.summary() # Step 5


X_opt = X [:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step3-4: Removing Adming spend = 60%, when our SL is 5%
regressor_OLS.summary() # Step 5

X_opt = X [:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #Step3-4: Removing Marketing spend = 6%, when our SL is 5%
regressor_OLS.summary() # Step 5


