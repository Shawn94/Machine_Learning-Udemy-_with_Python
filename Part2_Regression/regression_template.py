#Regression Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting the Regression Model to the dataset
#Create your regressor here

#Predicting a new result 
y_pred = regressor.predict(6.5)

#Visualising the Polynomial Regression results
plt.scatter(X,y color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the Polynomial Regression results(for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1) #gives curvy vector
X_grid = X_grid.reshape((len(X_grid),1)) #X_grid should be 2d matrix, reshape from vector -> matrix
plt.scatter(X,y color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()