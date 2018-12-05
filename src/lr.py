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

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

def tuneLR1(X,y):
    #1. everything is default
    clf = LogisticRegressionCV(cv=5, random_state=0,
                               multi_class='multinomial').fit(X, y)
    ##>>> clf.predict(X[:2, :])
    ##>>> clf.predict_proba(X[:2, :]).shape
    print( clf.score(X, y) )
    ## 0.5667011375387797
    
    #2. grid search for estimator
    print("test1")
    param_test1 = {'n_estimators':range(10,71,10)}
    gsearch1 = GridSearchCV(cv=None,
               estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
               param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
    gsearch1.fit(X,y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    ## {'C': 0.001} 0.5284384694932782
    """
    gsearch1 = GridSearchCV(cv=None,
               estimator=LogisticRegression(C=0.001, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
               param_grid={'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']})
    gsearch1.fit(X,y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    ##{'solver': 'newton-cg'} 0.8366080661840745
    """

    #3 
    print("test2")
    param_test2 = {'dual':range(False, True), 'fit_intercept':range(False, True)}
    gsearch2 = GridSearchCV(cv=None,
               estimator=LogisticRegression(C=0.001, intercept_scaling=1,   
               solver='newton-cg',penalty='l2', tol=0.0001),
               param_grid=param_test2)
    gsearch2.fit(X,y)
    print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
    ###{'dual': 0, 'fit_intercept': 0} 0.8474663908996898

    #4. 
    print("test3")
    param_test3 = {'intercept_scaling':range(1, 10,2)}
    gsearch3 = GridSearchCV(cv=None,
               estimator=LogisticRegression(C=0.001, solver='newton-cg',
               dual=False, fit_intercept=False,penalty='l2', tol=0.0001),
               param_grid=param_test3)
    gsearch3.fit(X,y)
    print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
    #{'intercept_scaling': 1} 0.8474663908996898

def tuneLR3(X,y):
    #1. everything is default
    clf = LogisticRegressionCV(cv=5, random_state=0,
                               multi_class='multinomial').fit(X, y)
    ##>>> clf.predict(X[:2, :])
    ##>>> clf.predict_proba(X[:2, :]).shape
    print( clf.score(X, y) )
    ## 0.9276111685625646
    
    #2. grid search for estimator
    print("test1")
    gsearch1 = GridSearchCV(cv=5, 
               estimator=LogisticRegression(intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
               param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
    gsearch1.fit(X,y)
    print(gsearch1.best_params_, gsearch1.best_score_)
    ## {'C': 0.001} 0.9110651499482937
    ##gsearch1 = GridSearchCV(cv=None,
    ##           estimator=LogisticRegression(C=0.001, intercept_scaling=1,   
    ##           dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
    ##           param_grid={'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']})
    ##gsearch1.fit(X,y)
    ##print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
    ##{'solver': 'newton-cg'} 0.8366080661840745

    #3 
    print("test2")
    param_test2 = {'dual':range(False, True), 'fit_intercept':range(False, True)}
    gsearch2 = GridSearchCV(cv=None,
                            estimator=LogisticRegression(C=0.001, intercept_scaling=1,   
                                ##solver='newton-cg',penalty='l2', tol=0.0001),
                                penalty='l2', tol=0.0001),
                            param_grid=param_test2)
    gsearch2.fit(X,y)
    print(gsearch2.best_params_, gsearch2.best_score_)
    ##{'dual': 0, 'fit_intercept': 0} 0.9188210961737332

    #4. 
    print("test3")
    param_test3 = {'intercept_scaling':range(1, 10,2)}
    gsearch3 = GridSearchCV(cv=None,
                            estimator=LogisticRegression(C=0.001, dual=False,
                                fit_intercept=False,penalty='l2', tol=0.0001),
                            param_grid=param_test3)
    gsearch3.fit(X,y)
    print( gsearch3.best_params_, gsearch3.best_score_)
    #{'intercept_scaling': 1} 0.9188210961737332

    #4. 
    print("test4")
    param_test4 = {'intercept_scaling':range(1, 10,2)}
    gsearch4 = GridSearchCV(cv=None,
                            estimator=LogisticRegression(C=0.001, dual=False,
                                fit_intercept=False,penalty='l2', tol=0.0001),
                            param_grid={'solver': ['newton-cg','lbfgs','liblinear','sag','saga']})
    gsearch4.fit(X,y)
    print( gsearch4.best_params_, gsearch4.best_score_)
    #{'solver': 'liblinear'} 0.9188210961737332


if __name__ == "__main__":
    column_name = 'adj_close_price'
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    sym_df = retrieve_id_ticker()

    df_a = make_dataset1(sym_df["ticker"][1], column_name,date1, date2)
    ## build the x_train and x_test
    #X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
    #X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
    #    y_train.to_csv("ytrain.csv",float_format='%.3f')
    #    y_test.to_csv("ytest.csv",float_format='%.3f')
    #tuneLR1(X_train,y_train)
    #The best parameters are {'C': 0.1, 'gamma': 1} with a score of 0.56
    #X_train, X_test, y_train, y_test = make_train_3way(df_a, column_name, 5, 5.5, train_date, test_date)
    #tuneLR3(X_train,y_train)

    ## this part try other models
    ##models = [("LR", LogisticRegression(C=10))]
    models = [("LR", LogisticRegression(C=0.1, dual=False, solver='newton-cg',
                                    fit_intercept=False,penalty='l2', tol=0.0001))
             ]

    for idx in range(1, 10):
        ticker = sym_df["ticker"][idx]

        df_a = make_dataset1(ticker, column_name,date1, date2)

        ## build the x_train and x_test
        ##X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
        ##X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
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



