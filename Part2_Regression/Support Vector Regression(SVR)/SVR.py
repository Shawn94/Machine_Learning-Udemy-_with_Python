#SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =  pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                 #Make object rfor each matrix
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_Y.fit_transform(y.reshape(-1,1)) # Reshape it as i`ts 2D array


#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)


y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))) #featureScale 6.5, then inverse regresssor

#Visualising the SVR results
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#Visualising the SVR results(for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1) #gives curvy vector
X_grid = X_grid.reshape((len(X_grid),1)) #X_grid should be 2d matrix, reshape from vector -> matrix
plt.scatter(X,y, color='red')
plt.plot(X_grid,regressor.predict(X_grid), color ='blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()