# Support Vector Regression

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
# Datos unicos en columnas
x = dataset.iloc[:, 1:2].values
# Datos unicos en columnas
y = dataset.iloc[:, 2:3].values

# Escalado de variables
# Para escalar los datos
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
# Escalando columna x
x = sc_x.fit_transform(x)
# Escalando columna y
y = sc_y.fit_transform(y)


# Ajustar la regresión con el dataset
from sklearn.svm import SVR
param=['rbf',17]
regression = SVR(*param)
regression.fit(x,y)

# Predicción de nuestros modelos
predecir=np.array([6.5,7.5,8.5])
predecir=predecir.reshape(-1,1)
# Escalar nuestros datos a predecir(No usar el fit_transform, porque ya esta ajustado y no debe tener otro ajuste)
predecir=sc_x.transform(predecir)
# Obtener las predicciones
y_pred = regression.predict(predecir)
# Transformacio inversa de un escalado
y_pred = sc_y.inverse_transform(y_pred)


# Visualización de los resultados del SVR
# Valores entre min y max del dataset con saltos de 0.1
x_grid=np.arange(x.min(),x.max(),0.1)
# Redimensionando ver el vector como una matriz de 90x1
x_grid=x_grid.reshape(-1,1)

plt.scatter(x, y, color = "red")
plt.scatter(predecir,regression.predict(predecir),color='green')
plt.plot(x_grid, regression.predict(x_grid), color = "blue")
plt.title("Modelo de Regresión SVR(Maquina de soporte vectorial)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


