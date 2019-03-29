#!/usr/bin/python
# encoding: utf-8
import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
from time import time

#####################################################################################################################################
### Tarefa 1: selecione quais recursos voce usara.
### features_list eh uma lista de strings, cada uma delas eh um nome de recurso.
### O primeiro recurso deve ser "poi".

print '############################## ETAPA 1 ##############################\n### Selecionar atributos\n'

# Escolha de quais recursos serao usados
choices = {'salary':			1,
           'deferral_payments':		1,
           'total_payments':		1,
           'loan_advances':		1,
           'bonus':			1,
           'restricted_stock_deferred':	1,
           'deferred_income':		1,
           'total_stock_value':		1,
           'expenses':			1,
           'exercised_stock_options':	1,
           'other':			1,
           'long_term_incentive':	1,
           'restricted_stock':		1,
           'director_fees':		1,
           'to_messages':		0,
           'from_poi_to_this_person':	0,
           'from_messages':		0,
           'from_this_person_to_poi':	0,
           'shared_receipt_with_poi':	0,
           'from_poi_percentage':	0,
           'to_poi_percentage':		0
           }

print 'Temos %d'%int(len(choices)-1), 'atributos disponiveis para escolher usar.\n\n'

print 'Foram escolhidos:\n'
# O primeiro recurso deve ser poi
features_list = ['poi']
for feature, choice in choices.items():
    if choice == 1:
        print feature
        features_list.append(feature)

print '\nAgora temos %d'%int(len(features_list)-1), 'de atributos no total.'
### Carrega o dicionario que contem o dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transformando em DataFrame
import pandas as pd
df = pd.DataFrame(data_dict).T

# Delcarando tudo float praticamente
df[['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']] = df[['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']].replace('NaN', np.NaN).astype('float')

# Mostrando os valores ausentes
print '\n\nValores ausentes em cada atributo:\n'
print df.isnull().sum().sort_values(ascending = False)
print '\nNumero de POIs:  ',df['poi'][df['poi'] == 1].count()
print 'Numero de n/POIs:',df['poi'][df['poi'] == 0].count()

#####################################################################################################################################
### Tarefa 2: Remover os outliers

print '\n\n############################## ETAPA 2 ##############################\n### Remover os outliers\n'

print 'Temos %d'%int(len(data_dict)), 'chaves no total.\n'
# Chave TOTALmente desnecessaria
print 'Chaves removidas por prejudicarem a analise:\n'
print 'Removido: TOTAL'
data_dict.pop('TOTAL', None)

# Remocao das chaves cujos dados faltantes excederem 16
i = 0
keys = []
for data in data_dict:
    missing=0
    for key, value in data_dict[data].items():
        if value == 'NaN':
            missing += 1
    i+=1
    if missing > 16:
        keys.append(data)
        
for key in keys:
    print 'Removido:',key
    data_dict.pop(key, None)
print '\nE depois da limpeza, ficamos com %d'%int(len(data_dict)), 'chaves no total.\n'

#####################################################################################################################################
### Tarefa 3: Criar novo recurso(s)
    
print '\n\n############################## ETAPA 3 ##############################\n### Criar novos recursos\n'

for key in data_dict:
    # Percentagem de mensagens enviadas da pessoa para POIs
    if (type(data_dict[key]['from_poi_to_this_person']) == str) or (type(data_dict[key]['to_messages']) == str):
        data_dict[key]['from_poi_percentage'] = 'NaN'
    else:
        data_dict[key]['from_poi_percentage'] = int((float(data_dict[key]['from_poi_to_this_person']) / float(data_dict[key]['to_messages'])) * 100)
    
    # Percentagem de mensagens recebidas de POIs para a pessoa    
    if (type(data_dict[key]['from_this_person_to_poi']) == str) or (type(data_dict[key]['from_messages']) == str):
        data_dict[key]['to_poi_percentage'] = 'NaN'
    else:
        data_dict[key]['to_poi_percentage'] = int((float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])) * 100)

print 'Foram criados dois novos atributos, mas estes\n nao foram escolhidos na etapa 1 por pussuire\n uma vantagem injusta.'

### Armazenar no my_dataset para facilitar a exploracao
my_dataset = data_dict

### Extrair recursos e rotulos do conjunto de dados para testes locais
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Dividir os dados
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# Se quiser mostrar as pontuacoes do KBest
from sklearn.feature_selection import SelectKBest
print '\n\nPontuacoes do KBest:\n'
for i,j in zip(features_list[1:], SelectKBest().fit(features_train, labels_train).scores_):
	print "%.2f <-"%j, i
	
#####################################################################################################################################
### Tarefa 4: Tente uma variedade de classificadores
### Por favor, nomeie seu classificador como clf para facilitar a exportacao abaixo.
### Observe que, se voce quiser fazer o PCA ou outras operacoes em varios estagios,
### sera necessário usar o Pipelines. Para mais informacoes:
### http://scikit-learn.org/stable/modules/pipeline.html
    
print '\n\n############################## ETAPA 4 ##############################\n### Treinar classificadores\n'

# Importando toda a brincadeira
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

# 1 --> MLPC
# 2 --> KNN
# 3 --> RandomForest
# 4 --> SVM
# 5 --> GPC
# 6 --> DecisionTree

# Opcao para selecionar o classificador
qual = 6

if qual == 1: # MLPC
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    print 'MLP Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('selector',     SelectKBest()),
                           ('resizer',      PCA(random_state=42)),
                           ('classifier',   MLPClassifier(random_state=42))])

    # Definindo os parametros de treino
    parameters = {'selector__k':                    [10, 12, 14, 'all'],    # 12
                  'resizer__svd_solver':            ['randomized'],         # randomized         
                  'resizer__n_components':          [4],                    # 4
                  'classifier__learning_rate_init': [0.01, 0.001],          # 0.001
                  'classifier__learning_rate':      ['constant'],           # constant
                  'classifier__alpha':              [0.01, 0.1],            # 0.01
                  'classifier__hidden_layer_sizes': [1, 2, 3],              # 2
                  'classifier__activation':         ['relu', 'logistic'],   # logistic
                  'classifier__solver':             ['adam'],               # adam
                  'classifier__max_iter':           [10]}                   # 10
elif qual == 2: # KNN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier
    print 'KNN Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('scaler',       None),
                           ('selector',     SelectKBest()),
                           ('resizer',      PCA(random_state=42)),
                           ('classifier',   KNeighborsClassifier())])

    # Definindo os parametros de treino
    parameters = {'scaler':                         [None, StandardScaler(),MinMaxScaler()],    # None
                  'selector__k':                    [10, 12, 14],                               # 14
                  'resizer__svd_solver':            ['randomized'],                             # randomized         
                  'resizer__n_components':          [4, 5],                                      # 5
                  'classifier__n_neighbors':        np.arange(2,5),                             # 3 
                  'classifier__p':                  np.arange(2,5)                              # 3
                  }
elif qual == 3: # RandomForest
    from sklearn.ensemble import RandomForestClassifier
    print 'RandomForest Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('selector',     SelectKBest()),
                           ('resizer',      PCA(random_state=42)),
                           ('classifier',   RandomForestClassifier(random_state=42))])

    # Definindo os parametros de treino
    parameters = {'selector__k':                    [6, 10, 14],        # 10
                  'resizer__svd_solver':            ['randomized'],     # randomized         
                  'resizer__n_components':          [4, 5],             # 4
                  'classifier__n_estimators':       [5, 10, 25],        # 5
                  'classifier__min_samples_split':  [5, 10],            # 10
                  'classifier__criterion':          ['gini', 'entropy'],# gini
                  'classifier__min_samples_leaf':   [2, 3],             # 2
                  'classifier__max_features':       ['auto']            # auto
                  }
elif qual == 4: # SVM
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.svm import SVC
    print 'SVM Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('scaler',       None),
                           ('selector',     SelectKBest()),
                           ('resizer',      PCA(random_state=42)),
                           ('classifier',   SVC(random_state=42))])

    # Definindo os parametros de treino
    parameters = {'scaler':                         [None, StandardScaler(),MinMaxScaler()],
                  'selector__k':                    [6, 10, 14],        # 6
                  'resizer__svd_solver':            ['randomized'],     # randomized         
                  'resizer__n_components':          [4, 5],             # 4
                  'classifier__C':                  [0.1, 1, 10, 100],  # 1 
                  'classifier__gamma':              [0.1, 1, 10, 100]   # 100
                  }
elif qual == 5: # GPC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.gaussian_process import GaussianProcessClassifier
    print 'GPC Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('scaler',       None),
                           ('selector',     SelectKBest()),
                           ('resizer',      PCA(random_state=42)),
                           ('classifier',   GaussianProcessClassifier(random_state=42))])

    # Definindo os parametros de treino
    parameters = {'scaler':                          [None, StandardScaler(),MinMaxScaler()],
                  'selector__k':                     [6, 10, 14],        # 14
                  'resizer__svd_solver':             ['randomized'],     # randomized         
                  'resizer__n_components':           [4, 5],             # 5
                  'classifier__n_restarts_optimizer':[1, 2, 3],          # 1
                  'classifier__max_iter_predict':    [1, 2, 3]           # 2
                  }
else: # DecisionTree
    from sklearn.tree import DecisionTreeClassifier
    print 'DecisionTree Classifier\n'
    
    # Definindo os passos da Pipeline
    pipe = Pipeline(steps=[('selector',   SelectKBest()),
                           ('resizer',    PCA(random_state=42)),
                           ('classifier', DecisionTreeClassifier(random_state=42))])

    # Definindo os parametros de treino
    parameters = {'selector__k':                   [6],                 # 6
                  'resizer__svd_solver':           ['randomized'],      # randomized
                  'resizer__n_components':         [4],                 # 4
                  'classifier__min_samples_split': [5, 10],             # 5
                  'classifier__criterion':         ['gini', 'entropy'], # gini
                  'classifier__min_samples_leaf':  [2, 3]}              # 3

# Definindo a validacao cruzada
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

# Pegando o horario nesse instante
t0 = time()

# Treinando a brincadeira toda
clf = GridSearchCV(pipe, parameters, scoring='f1', cv=sss).fit(features_train, labels_train)

# Mostrando o quao demorado foi treinar a brincadeira toda
print "\n\nTempo de treinamento:", round(time()-t0, 3), "s\n"

#####################################################################################################################################
### Tarefa 5: Ajuste seu classificador para obter uma precisao e revocacao
### superior a .3 usando nosso script de teste. Verifique o script tester.py
### na pasta final do projeto para obter detalhes sobre o metodo de avaliacao,
### especialmente a funcao test_classifier. Devido ao pequeno tamanho do
### conjunto de dados, o script usa validacao cruzada de divisao aleatoria
### estratificada. Para mais informacoes:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
print '\n\n############################## ETAPA 5 ##############################\n### Mostrar desempenho\n'

# Importando as metricas
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Criando as predicoes
pred = clf.predict(features_test)
predT = clf.predict(features_train)

### Mostrando o desempenho
print "Acuracia no teste:", accuracy_score(pred, labels_test)
print "Precisao no teste:", precision_score(pred, labels_test)
print "  Recall no teste:", recall_score(pred, labels_test)

print "\nAcuracia no treino:", accuracy_score(predT, labels_train)
print "Precisao no treino:", precision_score(predT, labels_train)
print "  Recall no treino:", recall_score(predT, labels_train),'\n\n'

# Melhores parametros
print 'Melhores parametros achados:\n'
print clf.best_params_

# Nota geral
print '\nScore:',clf.score(features_test, labels_test)

# Salvando o melhor ajuste dos parametros
clf = clf.best_estimator_

#####################################################################################################################################
### Tarefa 6: faca o despejo (dump) do classificador, do dataset e da feature_list
### para que todos possam verificar seus resultados. Voce nao precisa alterar nada
### abaixo, mas certifique-se de que a versao do poi_id.py que voce envia possa ser
### executada por conta propria e gere os arquivos .pkl necessarios para validar
### seus resultados.

print '\n\n############################## ETAPA 6 ##############################\n### Despejar as variaveis e realizar teste final\n'

dump_classifier_and_data(clf, my_dataset, features_list)

# Importando o tester.py pra testar mais rapido
from tester import *
print 'Avaliacao do tester.py:\n'
main()
