#Polynomial Regression
# Description: for X get data as matrix, not just as list
# no train and test set split is needed due to little dataset


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) #Depending on degree all magnitudes were multipled to itselff
                                   # fitting and transforming at once
line_reg2 = LinearRegression()
line_reg2.fit(X_poly,y)             #Apply SimpleLinearRegression

#Visualising the Linear Regression results
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X), color ='blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1) #Give more curvy lane for plot
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color = 'red')
plt.plot(X_grid,line_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

#Predicting a new result with Polynomial Regression
line_reg2.predict(poly_reg.fit_transform([[6.5]]))

