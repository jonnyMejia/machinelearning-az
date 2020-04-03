# Regresion Lineal Multiple

# Importar librerias
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Importar datasets
dataset= pd.read_csv('50_Startups.csv')
# Variables Independientes
x = dataset.iloc[:,:-1].values
# Variable Dependiente
y = dataset.iloc[:,-1].values 

# Preprocesar la data
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_transformer = Pipeline([
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
        ])
preprocesor_x = ColumnTransformer([
        ('cat',categorical_transformer,[3]),
        ('num',SimpleImputer(strategy='mean'),[0,1,2])
        ],remainder='passthrough')
x=preprocesor_x.fit_transform(x)

# Trampa de las variables dummy eliminar una variable dummy
x = x[:,1:]

# Dividir el dataset en entreneamiento y testing
from sklearn.model_selection import train_test_split
#0.8 indica que el 80% de los datos se usaran para el entrenamiento
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenmiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

# Prediccion de los resultados en el conjunto de testing
y_pred=regression.predict(x_test)

# Optimo de RLM utilizando la eliminacion hacia atras
# Aladiendo un vector de unos representando el termino independiente
import statsmodels.api as sm
x=np.append(arr=np.ones((x.shape[0],1)).astype(int), values=x,axis=1)


# Paso 1, se establece un nivel de significacion
SL = 0.05

# Paso 2,  Generar el modelo con todas las variables
x_opt = x[:,[0,1,2,3,4,5]]
regression_ols = sm.OLS(y,x_opt).fit()
regression_ols.summary()

# Paso 3, Elimnar la variable predictora con el p-valor mayor 
# al nivel de significacion mas grande, (X2 resulto tener mas p-valor)
x_opt = x[:,[0,1,3,4,5]]
regression_ols = sm.OLS(y,x_opt).fit()
regression_ols.summary()
# x1 tambien tiene mayor p-valor
x_opt = x[:,[0,3,4,5]]
regression_ols = sm.OLS(y,x_opt).fit()
regression_ols.summary()
# x4 tambien tiene mayor p-valor
x_opt = x[:,[0,3,5]]
regression_ols = sm.OLS(y,x_opt).fit()
regression_ols.summary()

# Automatizacion de la Regresion lineal multiple por eliminacion hacia atras

def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
SL = 0.05
x_opt = x[:,[0,1,2,3,4,5]]
X_Modeled = backwardElimination(x_opt, SL)

# Utilizando el R2 ajustado, donde debe acercarse al 1 para ser una mejor ajuste
def backwardElimination_r2(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


