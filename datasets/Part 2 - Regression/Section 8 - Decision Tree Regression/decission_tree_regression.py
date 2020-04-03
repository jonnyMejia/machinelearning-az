# Regresión con Árboles de Decisión

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor()
regression.fit(x,y)

# Predicción de nuestros modelos
y_pred = regression.predict([[6.5]])

# Visualización de los resultados del Modelo Polinómico
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(-1, 1)
plt.scatter(x, y, color = "red")
plt.plot(x, regression.predict(x), color = "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


