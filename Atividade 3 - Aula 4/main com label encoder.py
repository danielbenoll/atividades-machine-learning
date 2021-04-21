# importa as bibliotecas do pandas e numpy
import pandas as pd
import numpy as np

# Permite a vizualização total do csv que esta sendo utilizado para a leitura
np.set_printoptions(threshold=np.inf)

# a variavel base lê o arquivo 'hepatitis.csv'
base = pd.read_csv('hepatitis.csv')

# # utilizamos o método drop para excluir a linha onde houver uma '?' nas colunas presentes
base.drop(base[base.STEROID == '?'].index, inplace=True)
base.drop(base[base.FATIGUE == '?'].index, inplace=True)
base.drop(base[base.MALAISE == '?'].index, inplace=True)
base.drop(base[base.ANOREXIA == '?'].index, inplace=True)
base.drop(base[base.LiverBIG == '?'].index, inplace=True)
base.drop(base[base.LiverFIRM == '?'].index, inplace=True)
base.drop(base[base.SpleenPALPABLE == '?'].index, inplace=True)
base.drop(base[base.SPIDERS == '?'].index, inplace=True)
base.drop(base[base.ASCITES == '?'].index, inplace=True)
base.drop(base[base.VARICES == '?'].index, inplace=True)

# a variavel de 'previsores' servirá de base para se chegar a variável 'classe' que sera o resultado
previsores = base.iloc[:,0:13].values
classe = base.iloc[:, 13].values

# Sem a conversão do OneHotEncoder
from sklearn.preprocessing import LabelEncoder

LabelEncoder_previsores = LabelEncoder()

previsores[:,2] = LabelEncoder_previsores.fit_transform(previsores[:,2])
previsores[:,4] = LabelEncoder_previsores.fit_transform(previsores[:,4])
previsores[:,5] = LabelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = LabelEncoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = LabelEncoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = LabelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = LabelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = LabelEncoder_previsores.fit_transform(previsores[:,10])
previsores[:,11] = LabelEncoder_previsores.fit_transform(previsores[:,11])
previsores[:,12] = LabelEncoder_previsores.fit_transform(previsores[:,12])

# print('previsores sem one hot encoder:\n', previsores[0:10,:])

# não é necessário utilizar o label encoder na classe
# LabelEncoder_classe = LabelEncoder()
# classe = LabelEncoder_classe.fit_transform(classe)

# importa a biblioteca de escalonamento
from sklearn.preprocessing import StandardScaler

escalonamento= StandardScaler()

# Após o escalonamento, é substituido os novos dados pelos antigos pela variável 'previsores'
previsores = escalonamento.fit_transform(previsores)

# importa o método para fazer o treinamento de máquina
from sklearn.model_selection import train_test_split

# utiliza o método para fazer o treinamento utilizando 15% para teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size=0.15, random_state=0)

print('previsores_treinamento\n', len(previsores_treinamento))
print('previsores_teste\n', len(previsores_teste))
print('classe_treinamento\n', len(classe_treinamento))
print('classe_teste\n', len(classe_teste))

# 
# atividade anterior
# 

# importa o método de Gaussian naïve Bayes
from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

# Previsões calculadas em cima dos dados para teste

previsoes = classificador.predict(previsores_teste)

# importa o método de matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score

# variável responsável por falar a porcentagem de acerto do algoritmo
precisao = accuracy_score(classe_teste, previsoes)

# foi atribuido a variável a matriz de confusão
matriz= confusion_matrix(classe_teste, previsoes)

print('Algoritmo de ML com % de acerto de: {}%'.format(round(100*precisao,2)))
print('\nMatriz de erros e acertos:\n',matriz)
