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
from ind import *

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

def make_dataset1(ticker, start_date, end_date, start_test):

    column_name = 'adj_close_price'
    df_a = retrieve_daily_price(ticker, 
                                columns=('price_date', 'open_price','high_price','low_price', 'adj_close_price', 'volume'), 
                                start=start_date, end=end_date)
    ## df_a = retrieve_daily_price(ticker, columns=('price_date', 'Close'), start=start_date, end=end_date)
    ## moving_average(df_a, 15)
    df_a = momentum(df_a, 5, column_name)
    df_a = moving_average(df_a, 5, column_name)
    df_a = moving_average(df_a, 15, column_name)
    df_a = rate_of_change(df_a, 5, column_name)
    df_a = stochastic_oscillator_k(df_a, column_name)
    df_a = williams_r(df_a, 15, column_name)
    df_a = relative_strength_index(df_a, 15, column_name)
    df_a = price_volume_trend(df_a, 15, column_name)
    df_a.dropna(inplace=True)

    ##df_a = exponential_moving_average(df_a, 5, column_name)
    ##df_a = bollinger_bands(df_a, 5, column_name)
    ##print(df_a.shape[0], len(df_a.index))
    ##print(df_a.head())
    ##print(df_a.tail())
    return df_a

def tuneSVC(X,y):
    grid = GridSearchCV(SVC(),
                        param_grid={"C":[0.01, 0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
    grid.fit(X, y)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

if __name__ == "__main__":
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    sym_df = retrieve_id_ticker()

    """
    df_a = make_dataset1(sym_df["ticker"][1], date1, date2, test_date)
    ## build the x_train and x_test
    X_train = df_a[(df_a.index < test_date) & (df_a.index >= train_date)]
    X_test = df_a[df_a.index >= test_date]
    #X_train.to_csv("xtrain.csv",float_format='%.3f')
    #X_test.to_csv("xtest.csv",float_format='%.3f')
    
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret = pd.DataFrame(index=df_a.index)
    tsret["Today"] = df_a["adj_close_price"].pct_change(periods=5)*100.0
    tsret["Direction"] = np.sign(tsret["Today"])
    y = tsret["Direction"]
    y_train = y[(y.index < test_date) & (y.index >= train_date)]
    #y_train.loc[y_train.index[0], 'Direction'] = 1
    y_train.fillna(1.0, inplace=True)
    y_test = y[y.index >= test_date]
    #    y_train.to_csv("ytrain.csv",float_format='%.3f')
    #    y_test.to_csv("ytest.csv",float_format='%.3f')
    tuneSVC(X_train,y_train)
    #The best parameters are {'C': 0.1, 'gamma': 1} with a score of 0.56
    """

    ## this part try other models
    ##models = [("LR", LogisticRegression(C=10))]
    models = [("LDA", LDA()),
              ("QDA", QDA()),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0) )
             ]
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

    ##param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    param_grid = {'C': [0.001, 0.1, 1, 10, 100] }
    ##GridSearchCV(cv=None,
    ##         estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
    ##           dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
    ##         param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
    idx = 1
    while idx < 5:
        ticker = sym_df["ticker"][idx]

        df_a = make_dataset1(ticker, date1, date2, test_date)

        ## build the x_train and x_test
        X_train = df_a[(df_a.index < test_date) & (df_a.index >= train_date)]
        X_train.to_csv("xtrain.csv",float_format='%.3f')
        X_test = df_a[df_a.index >= test_date]
        X_test.to_csv("xtest.csv",float_format='%.3f')
    
        # Create the "Direction" column (+1 or -1) indicating an up/down day
        tsret = pd.DataFrame(index=df_a.index)
        tsret["Today"] = df_a["adj_close_price"].pct_change(periods=5)*100.0
        tsret["Direction"] = np.sign(tsret["Today"])
        y = tsret["Direction"]
        y_train = y[(y.index < test_date) & (y.index >= train_date)]
        #y_train.loc[y_train.index[0], 'Direction'] = 1
        y_train.fillna(1.0, inplace=True)
        y_test = y[y.index >= test_date]
        y_train.to_csv("ytrain.csv",float_format='%.3f')
        y_test.to_csv("ytest.csv",float_format='%.3f')

        
        # Iterate through the models
        for m in models:
        
            ##clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)

            ##m[1] = clf
            # Train each of the models on the training set
            m[1].fit(X_train, y_train)

            # Make an array of predictions on the test set
            pred = m[1].predict(X_test)

            # Output the hit-rate and the confusion matrix for each model
            print(ticker+ " "+m[0]+" Hit Rates/Confusion Matrices:\n")
            print("%s:%0.3f" % (m[0], m[1].score(X_train, y_train)))
            print("%s:%0.3f" % (m[0], m[1].score(X_test, y_test)))
            print("%s\n" % confusion_matrix(pred, y_test, labels=[1,-1]))

        idx += 1



