#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

print len(features_train[0])

smallset = False 

if smallset:
    features_train = features_train[:len(features_train)/10]
    features_test  = features_test[:len(features_test)/10]
    labels_train   = labels_train[:len(labels_train)/10]
    labels_test    = labels_test[:len(labels_test)/10] 

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print 'Training time:', round(time() - t0, 3), 's'
t0 = time()
pred = clf.predict(features_test)
print 'Predicting time:', round(time() - t0, 3), 's'
from sklearn.metrics import accuracy_score
score = accuracy_score(pred, labels_test)
print score


#########################################################


