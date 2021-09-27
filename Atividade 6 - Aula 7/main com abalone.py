# importa as bibliotecas do pandas
import pandas as pd

# a variavel base lê o arquivo 'abalone_sem_interrogacao.csv'
base = pd.read_csv('abalone_sem_interrogacao.csv')

# a variavel de 'previsores' servirá de base para se chegar a variável 'classe' que sera o resultado
previsores = base.iloc[:, 0:8].values
classe = base.iloc[:, 8].values

# importa métodos da biblioteca para poder transformar os valores de string em númerico
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder() # Atributos categóricos para numéricos:

# declara qual coluna está em string
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])

# o mesmo método só que para a variável classe

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# importa a biblioteca de escalonamento
from sklearn.preprocessing import StandardScaler # Escalonamento:
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# importa o método para fazer o treinamento de máquina
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# importa o método de KNeighbors
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# importa o método de matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print('Algoritmo de ML com % de Acerto de', round(100*precisao,2), '%')
print('\nMatriz de erros e acertos:\n', matriz)