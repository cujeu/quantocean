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

## execfile("/home/el/foo2/mylib.py")

if __name__ == "__main__":
    start_date = datetime.datetime(2008, 4, 1)
    end_date = datetime.datetime(2017, 5, 1)

    sym_df = retrieve_id_ticker()
    ticker = sym_df["ticker"][1]
    column_name = 'adj_close_price'

    df_a = retrieve_daily_price(ticker, 
                                columns=('price_date', 'open_price','high_price','low_price', column_name, 'volume'), 
                                start=start_date, end=end_date)
    ## df_a = retrieve_daily_price(ticker, columns=('price_date', 'Close'), start=start_date, end=end_date)
    ## moving_average(df_a, 15)
    ##df_a = momentum(df_a, 5, column_name)
    df_a = moving_average(df_a, 5, column_name)
    ##df_a = exponential_moving_average(df_a, 5, column_name)
    ##df_a = rate_of_change(df_a, 5, column_name)
    ##df_a = bollinger_bands(df_a, 5, column_name)
    ## df_a =  williams_r(df_a, 15, column_name)
    ##print(df_a.shape[0], len(df_a.index))
    print(df_a.head())
    print(df_a.tail())


