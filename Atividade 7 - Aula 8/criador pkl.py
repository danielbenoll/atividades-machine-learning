# importa as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

# Permite a vizualização total do csv que esta sendo utilizado para a leitura
np.set_printoptions(threshold=np.inf)

# a variavel base lê o arquivo 'hepatitis_sem_interrogacao.csv'
base_hepatitis = pd.read_csv('hepatitis_sem_interrogacao.csv')

X_hepatitis = base_hepatitis.iloc[:, 0:13].values
y_hepatitis = base_hepatitis.iloc[:, 13].values

# importa a biblioteca de escalonamento
from sklearn.preprocessing import StandardScaler

scaler_hepatitis = StandardScaler()
X_hepatitis = scaler_hepatitis.fit_transform(X_hepatitis)

# importa o método para fazer o treinamento de máquina
from sklearn.model_selection import train_test_split

X_hepatitis_treinamento, X_hepatitis_teste, y_hepatitis_treinamento, y_hepatitis_teste = train_test_split(X_hepatitis, y_hepatitis, test_size = 0.25, random_state = 0)

# abre/cria o 'hepatitis.pkl' no modo de escrita
with open('hepatitis.pkl', mode = 'wb') as f:
  pickle.dump([X_hepatitis_treinamento, y_hepatitis_treinamento, X_hepatitis_teste, y_hepatitis_teste], f)

