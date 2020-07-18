# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values # Writing '1:2' instead of '1' because weant X to be a matrix and not an array
y = dataset.iloc[:, 2:3].values

# Fitting Decision Tree regression to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result 
y_pred = regressor.predict(np.array([[6.5]]))

# Visualizing the Regression results (for higher resolution and correct graph)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth v Bluff {Decision Tree Regression}")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()