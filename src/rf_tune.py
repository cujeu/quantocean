#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from price_db import retrieve_id_ticker
from price_db import retrieve_daily_price
#from ind import *
from make_data import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

## tuning parameters for percentage change , 1:up -1: down
def tuneRF1(X,y):
    #1. everything is default
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X,y)
    print( rf0.oob_score_)
    #y_predprob = rf0.predict_proba(X)[:,1]
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)

    #2. grid search for estimator
    print("test1")
    param_test1 = {'n_estimators':range(10,71,10)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                            min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
                            ##param_grid = param_test1, scoring='roc_auc',cv=5)
                            param_grid = param_test1,cv=5)
    gsearch1.fit(X,y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    ## {'n_estimators': 20} 0.9984488107549121

    #3 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth
    print("test2")
    param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 20,
                            min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
                            param_grid = param_test2,iid=False, cv=5)
    gsearch2.fit(X,y)
    print(gsearch2.best_estimator_ , gsearch2.best_params_, gsearch2.best_score_)
    ###the hight score show {'min_samples_split': 90, 'max_depth': 5} 0.9989690721649485

    #4. 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_lea
    print("test3")
    param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 20, max_depth=5,
                            max_features='sqrt' ,oob_score=True, random_state=10),
                            param_grid = param_test3, iid=False, cv=5)
    gsearch3.fit(X,y)
    print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
    #{'min_samples_split': 100, 'min_samples_leaf': 20} 0.9984536082474227
    #

def tuneRF3old(X,y):
    #1. everything is default
    print("default test")
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X,y)
    print( rf0.oob_score_)
    #y_predprob = rf0.predict_proba(X)[:,1]
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)

    #2. grid search for estimator
    print("test1")
    param_test1 = {'n_estimators':range(10,180,20)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                            min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
                            ##param_grid = param_test1, scoring='roc_auc',cv=5)
                            param_grid = param_test1,cv=5)
    gsearch1.fit(X,y)
    print(gsearch1.cv_results_ ,gsearch1.best_estimator_ , gsearch1.best_params_, gsearch1.best_score_)
    ### {'n_estimators': 10} 0.781799379524302

    #3 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth
    print("test2")
    param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 10,
                            min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
                            param_grid = param_test2,iid=False, cv=5)
    gsearch2.fit(X,y)
    print(gsearch2.cv_results_ ,gsearch2.best_estimator_, gsearch2.best_params_, gsearch2.best_score_)
    ### {'min_samples_split': 50, 'max_depth': 5} 0.790019374668997

    #4. 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_lea
    print("test3")
    param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 10, max_depth=5,
                            max_features='sqrt' ,oob_score=True, random_state=10),
                            param_grid = param_test3, iid=False, cv=5)
    gsearch3.fit(X,y)
    print(gsearch3.cv_results_ ,gsearch3.best_estimator_, gsearch3.best_params_, gsearch3.best_score_)
    #{'min_samples_split': 80, 'min_samples_leaf': 50} 0.9286418256274269

## tuning parameters for 3way change , 1:up 0:no change -1: down
def tuneRF3(X,y):
    #1. everything is default
    print("default test")
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X,y)
    print( rf0.oob_score_, rf0.get_params())
    #y_predprob = rf0.predict_proba(X)[:,1]
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
    # 0.9513960703205792 {'criterion': 'gini', 'n_estimators': 10, 
    #                      'min_impurity_split': None, 'min_samples_split': 2, 
    #                      'bootstrap': True, 'min_impurity_decrease': 0.0, 
    #                      'max_features': 'auto', 'verbose': 0, 
    #                      'max_depth': None, 'warm_start': False, 
    #                      'min_weight_fraction_leaf': 0.0, 'n_jobs': None, 
    #                      'max_leaf_nodes': None, 'min_samples_leaf': 1, 'class_weight': None}

    #2. grid search for estimator
    print("test1")
    param_test1 = {'n_estimators':range(10,180,20)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(oob_score=True, random_state=10,
                                        criterion ='gini',min_samples_split=2,bootstrap = True,
                                        max_features='auto',min_samples_leaf=1 ),
                            param_grid = param_test1, cv=3)
    gsearch1.fit(X,y)
    print(gsearch1.best_estimator_ , gsearch1.best_params_, gsearch1.best_score_)
    ### {'n_estimators': 90} 0.7993795243019648

    """
    #3 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth
    print("test2")
    param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90,
                                        oob_score=True, random_state=10,
                                        criterion ='gini',bootstrap = True,
                                        max_features='auto',min_samples_leaf=1),
                            param_grid = param_test2,iid=False, cv=5)
    gsearch2.fit(X,y)
    print(gsearch2.cv_results_ ,gsearch2.best_estimator_, gsearch2.best_params_, gsearch2.best_score_)
    ### {'max_depth': 11, 'min_samples_split': 50} 0.7869452867463433

    #4. 划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_lea
    print("test3")
    param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90,max_depth=11,
                            oob_score=True, random_state=10),
                            param_grid = param_test3, iid=False, cv=5)
    gsearch3.fit(X,y)
    print(gsearch3.cv_results_ ,gsearch3.best_estimator_, gsearch3.best_params_, gsearch3.best_score_)
    #{'min_samples_leaf': 50, 'min_samples_split': 80} 0.9286418256274269
    """

if __name__ == "__main__":
    column_name = 'adj_close_price'
    """
    10 year date set
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    """
    """
    5 year date set
    """
    date1 = datetime.datetime(2010,1,1)  #(2008, 4, 1)
    date2 = datetime.datetime(2015,12,31) #(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2010,1,29) #(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2015,1,1) #(2016,1,1)
    sym_df = retrieve_id_ticker()

    df_a = make_dataset1(sym_df["ticker"][2], column_name,date1, date2)
    ## build the x_train and x_test and tuning parameters
    #X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
    #X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
    # tuneRF1(X_train,y_train)
    X_train, X_test, y_train, y_test = make_train_3way(df_a, column_name, 5, 5.5, train_date, test_date)
    tuneRF3(X_train,y_train)
    

    ## model and parameters for 3way change , 1:up 0 no change -1: down
    models = [
              ("RF3", RandomForestClassifier(n_estimators=90,oob_score=True, random_state=10))
              ##  ("RF3",RandomForestClassifier(n_estimators=90,max_depth=11, min_samples_split=80, 
              ##          min_samples_leaf=50, 
              ##          ##max_features='sqrt', bootstrap=True, oob_score=True, n_jobs=1, 
              ##          oob_score=True,random_state=10) )
             ]
    
    """
    ## model and parameters for percentage change , 1:up -1: down
    models = [
              ("RF1", RandomForestClassifier(
              	n_estimators=20, criterion='gini', 
                max_depth=5, min_samples_split=100, 
                min_samples_leaf=20, max_features='sqrt', 
                bootstrap=True, oob_score=True, n_jobs=1, 
                random_state=10, verbose=0) )
             ]
    """

    """
    ## in SVC C[less fit  -> over fit] 
    ##        gamma[#less support vector   -> more SV]
    models = [("LR", LogisticRegression()),
              ("LDA", LDA()), 
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
              	C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0) )
              ]
    """

    """
    ##param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    ##GridSearchCV(cv=None,
    ##         estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
    ##           dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
    ##         param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
    ##df_pred = pd.DataFrame(columns=['A'])
    for idx in range(1, 10):
        ticker = sym_df["ticker"][idx]

        df_a = make_dataset1(ticker, column_name,date1, date2)
        ## build the x_train and x_test
        ##X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
        ##X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
        X_train, X_test, y_train, y_test = make_train_3way(df_a, column_name, 5, 5.5, train_date, test_date)
        #X_train.to_csv(ticker+"_xtrain.csv",float_format='%.3f')
        #y_train.to_csv(ticker+"_ytrain.csv",float_format='%.3f')
        #X_test.to_csv(ticker+"_xtest.csv",float_format='%.3f')
        #y_test.to_csv(ticker+"_ytest.csv",float_format='%.3f')
        ##df_pred = y_test

        # Iterate through the models
        for m in models:
        
            # Train each of the models on the training set
            m[1].fit(X_train, y_train)

            # Make an array of predictions on the test set
            pred = m[1].predict(X_test)
            df_pred = pd.DataFrame(pred, index=y_test.index, columns=[ticker])
            
            #df_pred[ticker] = pred
            #se = pd.Series(pred)
            #df_pred[ticker] = se.values
            #print("pred shape:", pred.shape, 'last 5:', pred[-5:-1])

            # Output the hit-rate and the confusion matrix for each model
            #y_true = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2])
            #y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2])
            #Predicted  0  1  2  All
            #True                   
            #0          3  0  0    3
            #1          0  1  2    3
            #2          2  1  3    6    2(2 predicted as 0 for 2 times) 3(2 predicted as 2 for 3 times)
            #All        5  2  5   12    
            # 
            csvpath='pred'
            df_pred.to_csv(csvpath+'/'+ticker+'.csv')
            #df_pred.to_csv(os.path.join(csvpath,ticker+'.csv'))
            print(ticker+ " "+m[0]+" Hit Rates/Confusion Matrices:\n")
            print("%s:%0.3f" % (m[0], m[1].score(X_train, y_train)))
            print("%s:%0.3f" % (m[0], m[1].score(X_test, y_test)))
            print("%s\n" % confusion_matrix(pred, y_test))  ##, labels=[1,-1]))

    #df_pred.to_csv('pred.csv')
    """

