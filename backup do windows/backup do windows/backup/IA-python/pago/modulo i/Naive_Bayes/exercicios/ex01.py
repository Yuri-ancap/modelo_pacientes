#https://www.kaggle.com/dell4010/wine-dataset
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

dataframe = pd.read_csv('/home/yuri/backup do windows/backup do windows/backup/dados_para_ia/módulo-1/regressão/wine_dataset.csv')

faltantes = dataframe.isnull().sum()
percentual = 100*faltantes/len(dataframe['fixed_acidity'])
print(percentual) #dataset sem dados faltantess 


y = dataframe['style']
x = dataframe.drop('style', axis = 1)

normalizar = MinMaxScaler((0,1))
x_norm = normalizar.fit_transform(x)

modelo = GaussianNB()

x_treino, x_teste, y_treino, y_teste = train_test_split(x_norm, y, test_size = 0.3)

kfold = StratifiedKFold(5)

scores = cross_val_score(modelo, x_treino, y_treino, cv = kfold)

print(f"Scores de validação cruzada: {scores}")
print(f"Média dos scores de validação cruzada: {scores.mean()}")