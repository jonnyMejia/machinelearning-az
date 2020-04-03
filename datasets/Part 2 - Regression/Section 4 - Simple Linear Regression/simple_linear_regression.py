# IMPORTAR LIBRERIAS 
import pandas as ps
import matplotlib.pyplot as plt
import numpy as np

#Importando dataset de salarios
dataset = ps.read_csv('Salary_Data.csv')
# Variable Independiente (Años de experiencia)
x = dataset.iloc[:,:-1].values
# Variable Dependiente (Sueldo)
y = dataset.iloc[:,1].values

# No necesito rellenar valores vacios (SimpleImputer)
# No necesito usar (LabelEncoder,OneHotEncoder) para datos categoricos

# DIVIDIR LA DATA EN ENTRENAMIENTO Y TESTING
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

# Opcional usar el StandarScaler para la normalizacion
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

from  sklearn.linear_model import LinearRegression
regression = LinearRegression()
# Suministrar al modelo datos de entrenamiento
# Argumento normalize=True para normalizar 
regression.fit(x_train,y_train)

# Predecir el conjunto de Test
y_pred = regression.predict(x_test)

# Visualizar los resultados de entrenamiento
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, regression.predict(x_train),color='blue')
plt.title("Sueldo vs Años de experiencia(Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultasos de test
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regression.predict(x_train),color='blue')
plt.title("Sueldo vs Años de experiencia(Conjunto de Testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

