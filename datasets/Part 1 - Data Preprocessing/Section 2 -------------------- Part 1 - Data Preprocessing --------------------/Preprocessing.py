# -*- coding: utf-8 -*-
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
y=dataset.iloc[:,-1].values

##############################################################
# Importar librerias de tratamientos de datos
# Para escalar los datos
from sklearn.preprocessing import StandardScaler
# Para completar missing values
from sklearn.impute import SimpleImputer
# OneHotEncoder variables dummys, StandarScaler para normalizar, LabelEncoder para categorizar
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
# Pipeline sirve para unir transformadores
from sklearn.pipeline import Pipeline
# Column transformer asigna el transformador y una cierta columna
from sklearn.compose import ColumnTransformer

# Pipelines para unir transformadores y asignarlas a tipos numericos y categoricas
numeric_transform = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler()),
        ])
categorical_transform = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ])
# Transformador con ciertas columnas
preprocesor = ColumnTransformer([
        ('num',numeric_transform,[1,2]),        
        ('cat',categorical_transform,[0])        
        ])
x=preprocesor.fit_transform(x)

# Ya que la variable y solo tiene una columna no puedo utilizar ColumnTransformer
from sklearn.preprocessing import LabelEncoder
y[:,]=LabelEncoder().fit_transform(y[:,])

##############################################################

# Dividir el dataset en entreneamiento y testing
from sklearn.model_selection import train_test_split
#0.8 indica que el 80% de los datos se usaran para el entrenamiento
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

##############################################################




