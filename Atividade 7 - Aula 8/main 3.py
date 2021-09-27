# importa as bibliotecas
import pandas as pd
import numpy as np
import pickle

# Permite a vizualização total do csv que esta sendo utilizado para a leitura
np.set_printoptions(threshold=np.inf)

# abre o 'hepatitis.pkl' no modo de leitura
with open('hepatitis.pkl', 'rb') as f:  
  X_hepatitis_treinamento, y_hepatitis_treinamento, X_hepatitis_teste, y_hepatitis_teste = pickle.load(f)

# importa o método de Regressão Logistica    
from sklearn.linear_model import LogisticRegression

logistic_hepatitis = LogisticRegression(random_state=1)
logistic_hepatitis.fit(X_hepatitis_treinamento, y_hepatitis_treinamento)
Resultados = logistic_hepatitis.predict(X_hepatitis_teste)

# importa o método de Pontuação de Precisão
from sklearn.metrics import accuracy_score
Porcentagem_acerto = accuracy_score(y_hepatitis_teste, Resultados)
print('Algoritmo de ML com Acerto de:')

# importa a Matrix de Confusão
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(logistic_hepatitis)
cm.fit(X_hepatitis_treinamento, y_hepatitis_treinamento)
cm.score(X_hepatitis_teste, y_hepatitis_teste)