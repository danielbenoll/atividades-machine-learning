import pandas as pd
base = pd.read_csv('hepatitis_sem_interrogacao.csv')
previsores = base.iloc[:, 0:13].values
classe = base.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler # Escalonamento:
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)
# Antes do kNN
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
print('Algoritmo de ML com % de Acerto de', round(100*precisao,2), '%')
print('\nMatriz de erros e acertos:\n', matriz)