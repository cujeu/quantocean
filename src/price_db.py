#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import pandas as pd
import MySQLdb as mdb
#import pymysql as mdb
import datetime

def retrieve_id_ticker():
    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'password'
    db_name = 'securities_master'
    con = mdb.connect(db_host, db_user, db_pass, db_name,  charset="utf8")
    sql = """SELECT id, ticker FROM symbol AS sym
             ORDER BY sym.id ASC;"""
    data_df = pd.read_sql_query(sql, con, index_col='id')

    # Output the dataframe tail
    return data_df

def retrieve_daily_price(ticker='', columns=None, oder_column='price_date', index_col='price_date', start=None, end=None):
    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'password'
    db_name = 'securities_master'
    con = mdb.connect(db_host, db_user, db_pass, db_name,  charset="utf8")
    # Select all of the historic Google adjusted close data
    sql = """SELECT %s
                FROM symbol AS sym
                INNER JOIN daily_price AS dp
                ON dp.symbol_id = sym.id
                WHERE %s
                ORDER BY dp.%s ASC;"""

    if start is not None and end is not None:
        time_sql = "and dp.price_date between '%s' and '%s'" % (start, end)
    else:
        time_sql = ''

    where_sql = "sym.ticker = '%s' %s" % (ticker, time_sql)
    # print(where_sql)
    """
    printed sql: SELECT dp.price_date, dp.adj_close_price
                FROM symbol AS sym
                INNER JOIN daily_price AS dp
                ON dp.symbol_id = sym.id
                WHERE sym.ticker = 'AMZN' and dp.price_date between '2008-04-01 00:00:00' and '2017-05-01 00:00:00'
    """

    if columns is not None:
        columns_str = (("dp.%s, " * (len(columns) - 1)) + "dp.%s") % columns
    else:
        columns_str = '*'

    final_sql = sql % (columns_str, where_sql, oder_column)

    # print(final_sql)
    # Create a pandas dataframe from the SQL query
    data_df = pd.read_sql_query(final_sql, con=con, index_col=index_col)

    # Output the dataframe tail
    return data_df


if __name__ == "__main__":
    # print id and ticker by calling retrieve_id_ticker
    data = retrieve_id_ticker()
    print(data)

    """
    # print dataframe by calling retrieve_daily_price
    try:
        ticker = sys.argv[1] 
    except IndexError:
        print('usage: python retrieving_data.py TICKER')
        sys.exit(1)

    data = retrieve_daily_price(ticker, columns=('price_date', 'adj_close_price')
                         , start=datetime.datetime(2017, 4, 1)
                         , end=datetime.datetime(2017, 5, 1))
    print(data.tail())
    """

