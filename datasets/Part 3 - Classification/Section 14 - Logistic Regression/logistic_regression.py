# Regresión Logística

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Dividir el dataset en entreneamiento y testing
from sklearn.model_selection import train_test_split
#0.8 indica que el 80% de los datos se usaran para el entrenamiento
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

# Debido a que la edad y el sueldo toman valores muy diferentes edad[17-70] sueldo[10000-1000000]
# Debemos escalar la data x
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Ajustar el modelo de regression logistica
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# Prediccionde los resultados en el conjunto de testing
y_pred = classifier.predict(x_test)

# Elaborar un matriz de confusion, evalua la precision de un clasificador
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
"""
Usuarios que el algoritmo predijo que no debian comprar y compraron son 5
Usuarios que el algoritmo predijo que deberian comprar y compraron son 17
[[56,  2],
 [ 5, 17]]
"""

# Representacion Grafica en el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
# Meshgrid debuelve dos vslores que son el resultado de un cruze de los puntos x e y 
# x = {1,2}
# y = {2,3}
# x1 = [1,2],[1,2]  ;  x2 = [2,2],[3,3] como un plano cartesiano 
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
# establece un fondo uniendo las listas aplanadas ravel() y transpuestas T para cambiar el tamaño en el mismo de x1
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# establece el limite de las abcisa y ordenada x e y en sus limites maximos y minimos
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Pinta el color dependiendo de donde caiga y el fondo que tiene 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
# Muestra una leyenda de los puntos
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.70, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
              c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


