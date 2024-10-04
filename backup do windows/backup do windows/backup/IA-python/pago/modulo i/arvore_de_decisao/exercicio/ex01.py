import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

caminho = '/home/yuri/backup do windows/backup do windows/backup/dados_para_ia/módulo-1/decision_tree/archive/column_2C_weka.csv'
dataset = pd.read_csv(caminho)

#print(dataset.head(6)) 
#print(dataset.columns)
#print(f'\n\n {dataset.shape}')
n_linhas, n_colunas = dataset.shape

y = dataset['class']
x = dataset.drop('class', axis = 1)
y_corrigido = pd.Series(index = y.index, dtype = int)

zero_e_um = {'Abnormal':0, 'Normal':1}

for indice, binario in enumerate(y):
    y_corrigido[indice] = zero_e_um[binario] 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import StratifiedKFold
faltantes = dataset.isnull().sum()
percentual = 100*faltantes/n_linhas
print(faltantes)
print(f'{dataset.dtypes}')

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_corrigido, test_size = 0.3)

modelo = DecisionTreeClassifier()
kfold = StratifiedKFold(5) 

criterion = ['gini', 'entropy', 'log_loss']
max_depth = np.array([3, 4, 5, 6, 7, 9, 11])
min_samples_split = np.array([2, 3, 4, 5, 6, 7])

dicionario = {'criterion': criterion, 'max_depth': max_depth,'min_samples_split':min_samples_split}

treino = RandomizedSearchCV(modelo, param_distributions = dicionario, n_iter = 40, cv = kfold)
treino.fit(x_treino, y_treino)

print(f'acuracia: {treino.best_score_}')

