#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from price_db import retrieve_id_ticker
from price_db import retrieve_daily_price
from make_data import *

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

def tuneGBCT1(X,y):
    #1. everything is default
    #y_predprob = rf0.predict_proba(X)[:,1]
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)

    """
    #2. grid search for estimator
    print("test1")
    param_test1 = {'n_estimators':range(20,81,10)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                            min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                            param_grid = param_test1, iid=False,cv=5)
                            ###param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
    gsearch1.fit(X,y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    ## {'n_estimators': 30} 0.9989690721649485

    #3 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth/min_samples_split
    print("test2")
    param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, min_samples_leaf=20, 
                            max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test2, iid=False, cv=5)
    gsearch2.fit(X,y)
    print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
    ###the hight score show {'min_samples_split': 100, 'max_depth': 3} 0.9989690721649485
    #4. 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf
    print("test3")
    param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,
                            max_depth=3, max_features='sqrt', subsample=0.8, random_state=10), 
                            param_grid = param_test3, iid=False, cv=5)
    gsearch3.fit(X,y)
    print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
    #{'min_samples_leaf': 70, 'min_samples_split': 800} 0.9875754500293787
    
    #5. 对最大特征数max_features进行网格搜索
    print("test4")
    param_test4 = {'max_features':range(0,20,2)}
    gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,
                            max_depth=3, min_samples_leaf =70, 
                            min_samples_split =800, subsample=0.8, random_state=10), 
                            param_grid = param_test4, iid=False, cv=5)
    gsearch4.fit(X,y)
    print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)
    """
    #6. 子采样的比例进行网格搜索
    print("test5")
    param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,
                            max_depth=3, min_samples_leaf =70, 
                            min_samples_split =800, max_features=7, random_state=10), 
                            param_grid = param_test5, iid=False, cv=5)
    gsearch5.fit(X,y)
    print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
    # {'subsample': 0.6} 0.9989690721649485

if __name__ == "__main__":
    column_name = 'adj_close_price'
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    sym_df = retrieve_id_ticker()

    ## df_a = make_dataset1(sym_df["ticker"][1], column_name,date1, date2)
    ## build the x_train and x_test
    #X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
    #X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
    
    # tuneGBCT1(X_train,y_train)

    ## this part try other models
    models = [ ("GBCT", GradientBoostingClassifier(learning_rate=0.05, n_estimators=30, 
                        min_samples_leaf = 70, min_samples_split = 800, max_features=9, 
                        max_depth=3, subsample=0.6, random_state=10))
             ]

    for idx in range(1,10):
        ticker = sym_df["ticker"][idx]

        df_a = make_dataset1(ticker, column_name,date1, date2)

        ## build the x_train and x_test
        ##X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
        ## X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
        X_train, X_test, y_train, y_test = make_train_3way(df_a, column_name, 5, 5.5, train_date, test_date)
    
        # Iterate through the models
        for m in models:
        
            # Train each of the models on the training set
            m[1].fit(X_train, y_train)

            # Make an array of predictions on the test set
            pred = m[1].predict(X_test)
            print("pred shape:", pred.shape, 'last 5:', pred[-5:-1])

            # Output the hit-rate and the confusion matrix for each model
            print(ticker+ " "+m[0]+" Hit Rates/Confusion Matrices:\n")
            print("%s:%0.3f" % (m[0], m[1].score(X_train, y_train)))
            print("%s:%0.3f" % (m[0], m[1].score(X_test, y_test)))
            print("%s\n" % confusion_matrix(pred, y_test)) ##, labels=[1,-1]))




