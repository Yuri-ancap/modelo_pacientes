import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

x = pd.DataFrame(load_iris().data, columns = load_iris().feature_names)
y = pd.Series(load_iris().target, name = 'especie')

print(x.head())
print(y.head())
print(load_iris().keys())

#'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'


kfold = StratifiedKFold(5)

modelo = DecisionTreeClassifier()

treino1 = cross_val_score(modelo, x, y, cv = kfold)
#para normalizar 


normalizar = MinMaxScaler((0,1))
x_norm = normalizar.fit_transform(x)

treino2 = cross_val_score(modelo, x_norm, y, cv = kfold)
print(f'média sem normalização:{treino1.mean()}')

print(f'média com normalização:{treino2.mean()}')