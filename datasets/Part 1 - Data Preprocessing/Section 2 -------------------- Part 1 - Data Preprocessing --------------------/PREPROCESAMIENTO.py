# -*- coding: utf-8 -*-
"""
Preprocesado de datos completo
"""

# Importar librerias
# Contiene herramientas matematicas para realizar algoritmos de machine learning
import numpy as np
# Representacion grafica y dibujos en python
import matplotlib.pyplot as plt
# Carga de datos y manipular datos 
import pandas as pd

##############################################################

# Importar un data_set
dataset=pd.read_csv("Data.csv")
# Variable independiente(3 primeras columnas)
x= dataset.iloc[:,:-1].values
# Variable dependiente, la que yo quiero predecir
y=dataset.iloc[:,3].values

##############################################################

# Tratamiento de datos
from sklearn.impute import SimpleImputer
# Reemplazar por medias(mean,median,most_frequent)
imputer=SimpleImputer(strategy="mean",missing_values=np.nan)
# Reemplzar con medias en la columna con indice 1 y 2
x[:,1:3]=imputer.fit(x[:,1:3]).transform(x[:,1:3])

##############################################################

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder
x[:,0]=LabelEncoder().fit_transform(x[:,0])

##############################################################

#Variables Dummy, que añade mas columnas para evitar el orden en variables categoricas
from sklearn.preprocessing import OneHotEncoder
# primera columna, si necesito otras columas solo agregar un [0,3,4]
ohe=OneHotEncoder(categorical_features=[0])
x=ohe.fit_transform(x).toarray()
# Cuando solo son dos opciones solo hace falta un label encoder
y=LabelEncoder().fit_transform(y)

##############################################################

# Dividir el dataset en entreneamiento y testing
from sklearn.model_selection import train_test_split
#0.8 indica que el 80% de los datos se usaran para el entrenamiento
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

##############################################################

# Escalado de variables(Normalizar)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
# Se establece una transformacion
x_train=sc_x.fit_transform(x_train)
# Para que tenga la misma transofrmacion se usa solo transform
x_test=sc_x.transform(x_test)




