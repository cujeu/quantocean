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


"""
retrieve data from database and add some features
"""
def make_dataset1(ticker, column_name, start_date, end_date):

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

"""
build train/test data by 5 day lag, address change direction
"""
def make_train_dir(df_a, column_name, prd, p_train_date, p_test_date):
    ## build the x_train and x_test
    r_X_train = df_a[(df_a.index < p_test_date) & (df_a.index >= p_train_date)]
    r_X_test = df_a[df_a.index >= p_test_date]
    
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret = pd.DataFrame(index=df_a.index)
    tsret["Today"] = df_a[column_name].pct_change(periods=prd)*100.0
    tsret["Direction"] = np.sign(tsret["Today"])
    y = tsret["Direction"]
    r_y_train = y[(y.index < p_test_date) & (y.index >= p_train_date)]
    #y_train.loc[y_train.index[0], 'Direction'] = 1
    r_y_train.fillna(1.0, inplace=True)
    r_y_test = y[y.index >= p_test_date]

    return (r_X_train, r_X_test, r_y_train, r_y_test)

"""
build train/test data by 5 day lag, address change percentage
"""
def make_train_pct(df_a, column_name, prd, pct, p_train_date, p_test_date):
    ## build the x_train and x_test
    r_X_train = df_a[(df_a.index < p_test_date) & (df_a.index >= p_train_date)]
    r_X_test = df_a[df_a.index >= p_test_date]
    
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret = pd.DataFrame(index=df_a.index)
    tsret["Today"] = df_a[column_name].pct_change(periods=prd)*100.0 - pct 
    #tsret.to_csv("tsret2.csv",float_format='%.3f')

    tsret["Direction"] = np.sign(tsret["Today"])
    y = tsret["Direction"]
    r_y_train = y[(y.index < p_test_date) & (y.index >= p_train_date)]
    #y_train.loc[y_train.index[0], 'Direction'] = 1
    r_y_train.fillna(1.0, inplace=True)
    r_y_test = y[y.index >= p_test_date]

    return (r_X_train, r_X_test, r_y_train, r_y_test)

"""
build train/test data by 5 day lag, 0:no change 1:over 5% -1:under -5%
"""
def make_train_3way(df_a, column_name, prd, pct, p_train_date, p_test_date):
    ## build the x_train and x_test
    r_X_train = df_a[(df_a.index < p_test_date) & (df_a.index >= p_train_date)]
    r_X_test = df_a[df_a.index >= p_test_date]
    
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret = pd.DataFrame(index=df_a.index)
    tsret["Today"] = df_a[column_name].pct_change(periods=prd)*100.0 

    tsret["Direction"] = np.sign(tsret["Today"])

    i = 0
    while i < (len(tsret.index)-1):
        tsret.loc[tsret.index[i], 'Direction'] = 0
        if i >= prd:
            if (tsret.loc[tsret.index[i], 'Today'] >= pct):
                tsret.loc[tsret.index[i-prd], 'Direction'] = 1.0
            elif (tsret.loc[tsret.index[i], 'Today'] <= -pct):
                tsret.loc[tsret.index[i-prd], 'Direction'] = -1.0
        i = i + 1
    #tsret.to_csv("tsret2.csv",float_format='%.3f')


    y = tsret["Direction"]
    r_y_train = y[(y.index < p_test_date) & (y.index >= p_train_date)]
    #y_train.loc[y_train.index[0], 'Direction'] = 1
    r_y_train.fillna(1.0, inplace=True)
    r_y_test = y[y.index >= p_test_date]

    return (r_X_train, r_X_test, r_y_train, r_y_test)

if __name__ == "__main__":
    column_name = 'adj_close_price'
    date1 = datetime.datetime(2008, 4, 1)
    date2 = datetime.datetime(2017, 5, 1)
    ## train_date = datetime.datetime(2008, 4, 22) # one day ahead
    train_date = datetime.datetime(2008, 4, 29)  # 5 day ahead
    test_date = datetime.datetime(2016,1,1)
    sym_df = retrieve_id_ticker()

    df_a = make_dataset1(sym_df["ticker"][1], column_name, date1, date2)
    ## build the x_train and x_test
    ## X_train, X_test, y_train, y_test = make_train_dir(df_a, column_name, 5, train_date, test_date)
    ## X_train, X_test, y_train, y_test = make_train_pct(df_a, column_name, 5, 5.5, train_date, test_date)
    ##X_train, X_test, y_train, y_test = make_train_3way(df_a, column_name, 5, 5.5, train_date, test_date)
    
    ##X_train.to_csv("xtrain.csv",float_format='%.3f')
    ##X_test.to_csv("xtest.csv",float_format='%.3f')
    ##y_train.to_csv("ytrain.csv",float_format='%.3f')
    ##y_test.to_csv("ytest.csv",float_format='%.3f')

