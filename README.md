# Projeto de Machine Learning: Identificar fraudes nos e-mails da Enron - [Udacity](https://www.udacity.com/)

## Introdução
Em 2000, Enron era uma das maiores empresas dos EUA. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros dos executivos nos mais altos níveis da empresa. Caso haja mais interesse por trás desse caso, existe um [documentário](https://en.wikipedia.org/wiki/Enron:_The_Smartest_Guys_in_the_Room) referente a essa grande ascensão e ruptura da empresa.

Esse projeto visou praticar as habilidades de machine learning durante a construção de um modelo preditivo para tentar determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa.

### Objetivos
* Lidar com um conjunto de dados real e suas imperfeições;
* Validar resultados de aprendizagem de máquina usando dados de teste;
* Avaliar resultados de aprendizagem de máquina usando métricas quantitativas;
* Criar, selecionar e transformar atributos;
* Comparar a performance de algoritmos de aprendizagem de máquina;
* Otimizar algoritmos de aprendizagem de máquina para obter máxima performance;
* Comunicar resultados de aprendizagem de máquina de forma clara.

## Desenvolvimento do projeto

### Recursos necessários
Baixado o projeto incial do [repositório da Udacity](https://github.com/udacity/ud120-projects), cujo código pode ser encontrado no diretório `final_project`. Alguns arquivos relevantes são:

`poi_id.py`: Código inicial do identificar de pessoas de interesse (POI, do inglês Person of Interest). É neste arquivo que foi escrita a análise.

`final_project_dataset.pkl`: O conjunto de dados para o projeto. Veja mais detalhes abaixo. 

`tester.py`: Ao enviar o algoritmo, conjunto de dados, e a lista de atributos que foram utilizados (criados automaticamente pelo arquivo poi_id.py) para análise da Udacity. O avaliador usou este código para testar os resultados, para garantir que a performance é similar a obtida no relatório. Não precisa usar/modificar este código, mas é fornecido de forma transparente para os alunos testem seus algoritmos. 

`emails_by_address`: Este diretório contém diversos arquivos de texto, cada um contendo todas as mensagens de ou para um endereço de email específico. Estes dados estão aqui para referência, ou caso deseje criar atributos mais complexos baseando-se nos detalhes dos emails. A utilização desse arquivo é opcional no projeto.

### Etapas
Fornecido um código inicial pela Udacity que carrega os dados, seleciona os atributos e os colocam em um vetor `numpy`. O trabalho a partir disso foi utilizar engenharia sobre os atributos, escolher, otimizar algoritmos e testar suas respectivas capacidades preditivas. Depois será apresentado o desempenho dos algoritmos pós otimização.

Como etapa de pré-processamento deste projeto, foi combinado os dados da base "Enron email and financial" em um dicionário, onde cada par chave-valor corresponde a uma pessoa. A chave do dicionário é o nome da pessoa, e o valor é outro dicionário, que contém o nome de todos os atributos e seus valores para aquela pessoa. Os atributos nos dados possuem basicamente três tipos: atributos financeiros, de e-mail e rótulos POI (pessoa de interesse).

**atributos financeiros**: \['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))

**atributos de email**: \['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)

**rótulo POI**: \[‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro)

Encorajado a criar, transformar e re-escalar novos atributos a partir dos originais. Que deveriam ser armazenenadoos na estrutura `my_dataset`, e se utilizado estes atributos no modelo final, chama-los de `my_feature_list`, para que o avaliador fosse capaz de acessá-la durante os testes.

Como parte do projeto submetido, foi respondido uma série de perguntas demonstrando a linha de pensamento, que de certa forma é mais importante que o projeto final em si. Essa perguntas e respostas podem ser encontradas no arquivo `Questionário`.
