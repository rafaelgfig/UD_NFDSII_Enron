#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### Reduz a base de dados de treino para 1%
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf = SVC(kernel='rbf',C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print "tempo de treinamento:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao:", round(time()-t1, 3), "s"
print "acuracia:", accuracy_score(labels_test, pred)
print "10o", pred[10]
print "260", pred[26]
print "50o", pred[50]
print pred.sum()
#########################################################


