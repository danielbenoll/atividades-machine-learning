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

print('Parâmetro B0:', logistic_hepatitis.intercept_)
print('Parâmetros B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12 (os atributos):', logistic_hepatitis.coef_)