import pandas as pd
import numpy as np
import pylab as plt
import os
import sklearn.metrics
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


#classify
#train on set1

DataMatrix_set1 = pd.read_csv('Data_S1.csv')
DataMatrix_set2 = pd.read_csv('Data_S2.csv')

trainingset=DataMatrix_set1[DataMatrix_set1['set']=='train']
X = trainingset.iloc[:,2:-4].values
y = trainingset.iloc[:,-1].values

testset=DataMatrix_set1[DataMatrix_set1['set']=='test']
X_test = testset.iloc[:,2:-4].values
y_test = testset.iloc[:,-1].values

#first round of hyperparameter tuning
gammas = np.array([1e-5,1e-6,1e-7,1e-8])
Cs = np.array([1e5,1e6,1e7,1e8])
parameter_ranges = {'gamma':gammas,'C':Cs}
svc = SVC(kernel='rbf')
svc_search = GridSearchCV(svc, parameter_ranges, cv=3)
svc_search.fit(X,y)
print(svc_search.best_estimator_, svc_search.best_score_)

#second round of hyperparameter tuning
scale=[0.1,0.25,0.5,0.75,1,2.5,5,7.5,10]
gammas_rd2 = [i * svc_search.best_estimator_.gamma for i in scale]
Cs_rd2 = [i * svc_search.best_estimator_.C for i in scale]
recall = make_scorer(recall_score)
parameter_ranges = {'gamma':gammas_rd2,'C':Cs_rd2}
svc_rd2 = SVC(kernel='rbf')
svc_search_rd2 = GridSearchCV(svc_rd2, parameter_ranges, cv=3, scoring = recall)
svc_search_rd2.fit(X,y)
print(svc_search_rd2.best_estimator_)
print(svc_search_rd2.best_score_)

y_predict = svc_search_rd2.best_estimator_.predict(X)
cm1 = confusion_matrix(y, y_predict)

y_test_predict = svc_search_rd2.best_estimator_.predict(X_test)
cm2 = confusion_matrix(y_test, y_test_predict)

#classification for set 2
X_set2 = DataMatrix_set2.iloc[:,2:-3].values
y_set2 = DataMatrix_set2.iloc[:,-1].values
y_predict_set2 = svc_search_rd2.best_estimator_.predict(X_set2)
cm3 = confusion_matrix(y_set2, y_predict_set2)