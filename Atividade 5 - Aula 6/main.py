# importa as bibliotecas Orange e numpy
import Orange
import numpy as np

# Permite a vizualização total do csv que esta sendo utilizado para a leitura
np.set_printoptions(threshold=np.inf)

# a variavel base lê o arquivo 'hepatitis_sem_interrogacao.csv'
base = Orange.data.Table('hepatitis_sem_interrogacao.csv')

## variavel vai receber o método para aprender as regras
# cn2_learner = Orange.classification.rules.CN2Learner()
# classificador = cn2_learner(base)

## Visualizar as regras criadas:
# for regras in classificador.rule_list:
#   print(regras)

## Correspondência do resultado para teste
# AGE = 50, SEX = 1, STEROID = 1, ANTIVIRALS = 2, FATIGUE = 1, MALAISE = 2, ANOREXIA = 2, LiverBIG = 1, LiverFIRM = 2, SpleenPALPABLE = 2, SPIDERS = 2, ASCITES = 2, VARICES = 2
# AGE = 51, SEX = 1, STEROID = 1, ANTIVIRALS = 2, FATIGUE = 1, MALAISE = 2, ANOREXIA = 1, LiverBIG = 2, LiverFIRM = 2, SpleenPALPABLE = 1, SPIDERS = 1, ASCITES = 2, VARICES = 2

# resultado = classificador([['50', '1', '1', '2', '1', '2', '2', '1', '2', '2', '2', '2', '2'], ['51', '1', '1', '2', '1', '2', '1', '2', '2', '1', '1', '2', '2']])
# for i in resultado:
#   print('Resultado:', base.domain.class_var.values[i])

# a variavel base_dividida, vai fazer um array com 2 indexes onde um array será para 75% treinamento e outra para 25% teste
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

# print('tamanho da base Original: ',len(base))
# print('\ntamanho da base de Treinamento: ',len(base_treinamento))
# print('\ntamanho da base de Teste: ',len(base_teste), '\n')

# variavel vai receber o método para aprender as regras
cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: classificador])
print(Orange.evaluation.CA(resultado))
print('Porcentagem de acerto:', round(100*Orange.evaluation.CA(resultado)[0],2),'%')