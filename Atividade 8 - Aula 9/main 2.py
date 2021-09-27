# importa a biblioteca de SVC
from sklearn.svm import SVC
import pickle

# abre o 'hepatitis.pkl' no modo de leitura
with open('hepatitis.pkl', 'rb') as f:  
  X_hepatitis_treinamento, y_hepatitis_treinamento, X_hepatitis_teste, y_hepatitis_teste = pickle.load(f)

# utilizamos o método de SVC dando como parâmetro o C = que é a punição por classificação incorreta, juntamente com o kernel "linear"
svm_hepatitis = SVC(kernel='rbf', random_state=1, C = 10.0) # Testar com Kernels e C (2 a 4)
svm_hepatitis.fit(X_hepatitis_treinamento, y_hepatitis_treinamento)
previsoes = svm_hepatitis.predict(X_hepatitis_teste)

# importando a biblioteca de Matriz de Confusão
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svm_hepatitis)
cm.fit(X_hepatitis_treinamento, y_hepatitis_treinamento)
cm.score(X_hepatitis_teste, y_hepatitis_teste)