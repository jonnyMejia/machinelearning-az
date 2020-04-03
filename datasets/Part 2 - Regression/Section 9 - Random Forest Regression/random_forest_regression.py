#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:53:04 2019

@author: juangabriel
"""

# Regresión Bosques Aleatorios

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
params={"n_estimators":300,"random_state":0,"max_features":"auto"}
regression = RandomForestRegressor(**params)
regression.fit(x, y)

# Predicción de nuestros modelos con Random Forest
y_pred = regression.predict([[6.5]])

# Visualización de los resultados del Random Forest
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regression.predict(x_grid), color = "blue")
plt.title("Modelo de Regresión con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


