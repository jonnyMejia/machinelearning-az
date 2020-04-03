# Regresion Polinomica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Solo usaremos el Level para estimar salary
dataset=pd.read_csv("Position_Salaries.csv")
# Esto sera un vector de la variable x
x=dataset.iloc[:,1].values
# Esto sera una matriz de la variable x
x=dataset.iloc[:,1:2].values
# Esto sera una matriz de la variable y
y=dataset.iloc[:,2:3].values
# Tener como matriz a pesar que sea un vector, me ayudara a no tener un errror de shape()

# No usaremos la division en entrenamiento y testing por que solo tenemos 10 datos

# No debemos usar la escalizacion ya que no haremos una lineal


# Ajustar la regrssion polinomico con el dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Establecer el grado del polinomio
poly_reg=PolynomialFeatures(degree=3)
# Obtener sus respectivos grados(x^2,x^3,x^4,...) de cada dato de la matriz
x_poly=poly_reg.fit_transform(x)
# Usar la clase linearRegression y ajustarla a ala variable y 
lin_reg_pol= LinearRegression()
lin_reg_pol.fit(x_poly,y)


# Visualizacion de los resultados del modelo Polinomico
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_pol.predict(x_poly),color='blue')
plt.title('Modelo de regresion Polinomial')
plt.ylabel('Sueldo (en $)')
plt.xlabel('Posicion del empleado')
plt.show()
 
# Visualizacion de los resultados pero mas suavizados
# Valores entre min y max del dataset con saltos de 0.1
x_grid=np.arange(x.min(),x.max(),0.1)
# Redimensionando ver el ector como una matriz de 90x1
x_grid=x_grid.reshape(-1,1)

plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_pol.predict(poly_reg.transform(x_grid)),color='blue')
plt.title('Modelo de regresion Polinomial')
plt.ylabel('Sueldo (en $)')
plt.xlabel('Posicion del empleado')
plt.show()


# Prediccion de nuestros modelos 
# Un trabajador quisiera entrar en tre el nivel 6 y 7 y queremos predecir cuando ganaria
predecir=np.array([6.5,7.5,8.5])
predecir=predecir.reshape(-1,1)
lin_reg_pol.predict(poly_reg.fit_transform(predecir))

