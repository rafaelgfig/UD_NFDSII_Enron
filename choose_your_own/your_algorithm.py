#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import sys
from time import time
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# "Bonus"
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# "Bonus do Bonus"
from sklearn.neural_network import MLPClassifier

######################################################################
name = "KNeighborsClassifier"
clf = KNeighborsClassifier()
clf.fit(features_train, labels_train)

t0 = time()
clf.fit(features_train, labels_train)
print "\n"
print "tempo de treinamento do KNN:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do KNN:", round(time()-t1, 3), "s"

print "acuracia do KNN:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass

######################################################################
name = "RandomForestClassifier"
clf = RandomForestClassifier()
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do RandomForest:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do RandomForest:", round(time()-t1, 3), "s"

print "acuracia do RandomForest:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass
	
######################################################################
name = "AdaBoostClassifier"
clf = AdaBoostClassifier()
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do Ada:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do Ada:", round(time()-t1, 3), "s"

print "acuracia do Ada:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass
	
###################################################################### THE BEST
name = "SVC"
clf = SVC(kernel='rbf', C=10000.0, gamma=1)
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do SVM:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do SVM:", round(time()-t1, 3), "s"

print "acuracia do SVM:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass

######################################################################
name = "DecisionTreeClassifier"
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do DecisionTree:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do DecisionTree:", round(time()-t1, 3), "s"

print "acuracia do DecisionTree:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass

######################################################################
name = "GaussianNB"
clf = GaussianNB()
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do NaiveBayes:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do NaiveBayes:", round(time()-t1, 3), "s"

print "acuracia do NaiveBayes:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass
	
######################################################################
name = "MLPClassifier"
clf = MLPClassifier()
clf.fit(features_train, labels_train)

t0 = time()
clf = clf.fit(features_train, labels_train)
print "tempo de treinamento do MLPClassifier:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de predicao do MLPClassifier:", round(time()-t1, 3), "s"

print "acuracia do MLPClassifier:", accuracy_score(labels_test, pred)
print "\n"
try:
    prettyPicture(clf, features_test, labels_test, name)
except NameError:
    pass