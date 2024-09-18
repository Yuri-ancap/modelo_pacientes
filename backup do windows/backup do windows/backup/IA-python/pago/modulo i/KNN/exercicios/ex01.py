from sklearn.datasets import load_iris
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
iris = load_iris()

x = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target)

from sklearn.model_selection import RandomizedSearchCV 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

normalizador = MinMaxScaler((0, 1))
x_norm = normalizador.fit_transform(x)

modelo = KNeighborsClassifier()

valores_k = np.array([3, 5, 7, 9, 11])
valores_p = ['minkowski', 'chebyshev']
distancia = np.array([1, 2, 3, 4])

parametros = {'n_neighbors': valores_k, 'p': distancia, 'metric': valores_p}

treino = RandomizedSearchCV(estimator = modelo, param_distributions = parametros, n_iter = 50, cv=5)

treino.fit(x_norm, y)

print(f"Melhores parâmetros: {treino.best_params_}")
print(f"Melhor acurácia: {treino.best_score_}")
