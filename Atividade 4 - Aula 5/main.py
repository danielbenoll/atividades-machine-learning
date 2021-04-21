# importa as bibliotecas do pandas e numpy
import pandas as pd
import numpy as np

# Permite a vizualização total do csv que esta sendo utilizado para a leitura
np.set_printoptions(threshold=np.inf)

# a variavel base lê o arquivo 'abalone.csv'
base = pd.read_csv('abalone.csv')

# a variavel de 'previsores' servirá de base para se chegar a variável 'classe' que sera o resultado
previsores = base.iloc[:,0:8].values
classe = base.iloc[:, 8].values

# importa métodos da biblioteca para poder transformar os valores de string em númerico
from sklearn.preprocessing import LabelEncoder

# Transforma string em int
LabelEncoder_previsores = LabelEncoder()
previsores[:,0] = LabelEncoder_previsores.fit_transform(previsores[:,0])

# importa a biblioteca de escalonamento
from sklearn.preprocessing import StandardScaler

escalonamento= StandardScaler()

# Após o escalonamento, é substituido os novos dados pelos antigos pela variável 'previsores'
previsores = escalonamento.fit_transform(previsores)

# importa o método para fazer o treinamento de máquina
from sklearn.model_selection import train_test_split

# utiliza o método para fazer o treinamento utilizando 15% para teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size=0.15, random_state=0)

# importa a biblioteca de arvores de decisão
from sklearn.tree import DecisionTreeClassifier

classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)

# calcula as previsões usando de base os dados de teste
previsoes = classificador.predict(previsores_teste)

# importa o método de matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score

# variável responsável por falar a porcentagem de acerto do algoritmo
precisao = accuracy_score(classe_teste, previsoes)

# foi atribuido a variável a matriz de confusão
matriz= confusion_matrix(classe_teste, previsoes)

print('Algoritmo de ML com % de acerto de: {}%'.format(round(100*precisao,2)))
print('\nMatriz de erros e acertos:\n',matriz)
