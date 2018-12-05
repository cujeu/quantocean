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
#from ind import *
from make_data import *

from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    column_name = 'adj_close_price'
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    sym_df = retrieve_id_ticker()

    ## this part try other models
    models = [
              ("LDA", LDA())
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

    for idx in range(1, 5):
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

            # Make numpy array of predictions on the test set
            pred = m[1].predict(X_test)
            print("pred shape:", pred.shape, 'last 5:', pred[-5:-1])

            # print(pred)
            # pred looks like
            # [ 0.  0. -1. -1. -1.  0.  0.  0.  1.  0.  0. -1.  0.  0.  0.  0.  0.  0.
            #  0. -1.  0.  0.  1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.  0.  0.
            #  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]

            # Output the hit-rate and the confusion matrix for each model
            print(ticker+ " "+m[0]+" Hit Rates/Confusion Matrices:\n")
            print("%s:%0.3f" % (m[0], m[1].score(X_train, y_train)))
            print("%s:%0.3f" % (m[0], m[1].score(X_test, y_test)))
            print("%s\n" % confusion_matrix(pred, y_test)) ##, labels=[1,0,-1]))



