#!/usr/bin/python
# -*- coding: utf-8 -*-

# price_retrieval.py
#!/usr/bin/python
# -*- coding: utf-8 -*-

# price_retrieval.py

from __future__ import print_function

import datetime
import warnings
from six.moves.urllib.parse import urlencode

import MySQLdb as mdb
import requests
import quandl

# Obtain a database connection to the MySQL instance
db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'password'
db_name = 'securities_master'
con = mdb.connect(db_host, db_user, db_pass, db_name)


def obtain_list_of_db_tickers():
    """
    Obtains a list of the ticker symbols in the database.
    """
    with con:
        cur = con.cursor()
        cur.execute("SELECT id, ticker FROM symbol")
        data = cur.fetchall()
        return [(d[0], d[1]) for d in data]

def get_daily_historic_data_google(
        ticker, start_date=datetime.datetime(year=2008, month=1, day=1).date,
        end_date=datetime.date.today()
    ):
    """
    Obtains data from Google Finance returns and a list of tuples.
    ticker: Google Finance ticker symbol, e.g. "GOOG" for Google, Inc.
    start_date: Start date in (YYYY, M, D) format
    end_date: End date in (YYYY, M, D) format
    """
    # Construct the Google URL with the correct integer query parameters
    # for start and end dates. Note that some parameters are zero-based!
    params = {
            ## 'q':ticker, 'startdate':start_date.strftime('%Y-%m-%d'),'enddate':end_date.strftime('%Y-%m-%d'), 'output':'csv'
            'q':ticker, 'startdate':start_date.strptime('%Y-%m-%d'),
            'enddate':end_date.strptime('%Y-%m-%d'), 'output':'csv'
    }
    url = "http://finance.google.com/finance/historical?"
    url = url + urlencode(params)

    # Try connecting to Google Finance and obtaining the data
    # On failure, print an error message.
    try:
        yf_data = requests.get(url).text.split("\n")[1:-1]
        prices = []
        for y in yf_data:
            p = y.strip().split(',')
            prices.append(
                (datetime.datetime.strptime(p[0], '%Y-%m-%d'),
                p[1], p[2], p[3], p[4], p[5], p[6])
            )
    except Exception as e:
        print("Could not download Yahoo data: %s" % e)
    return prices


def insert_daily_data_into_db(
        data_vendor_id, symbol_id, daily_data
    ):
    """
    Takes a list of tuples of daily data and adds it to the
    MySQL database. Appends the vendor ID and symbol ID to the data.
    daily_data: List of tuples of the OHLC data (with
    adj_close and volume)
    """
    # Create the time now
    now = datetime.datetime.utcnow()

    # Amend the data to include the vendor ID and symbol ID
    daily_data = [
        (data_vendor_id, symbol_id, d[0], now, now,
        d[1], d[2], d[3], d[4], d[5], d[6])
        for d in daily_data
    ]

    # Create the insert strings
    column_str = """data_vendor_id, symbol_id, price_date, created_date,
                 last_updated_date, open_price, high_price, low_price,
                 close_price, volume, adj_close_price"""
    insert_str = ("%s, " * 11)[:-2]
    final_str = "INSERT INTO daily_price (%s) VALUES (%s)" % \
        (column_str, insert_str)

    # Using the MySQL connection, carry out an INSERT INTO for every symbol
    with con:
        cur = con.cursor()
        cur.executemany(final_str, daily_data)


if __name__ == "__main__":
    # This ignores the warnings regarding Data Truncation
    warnings.filterwarnings('ignore')

    # Loop over the tickers and insert the daily historical
    # data into the database
    tickers = obtain_list_of_db_tickers()
    lentickers = len(tickers)
    for i, t in enumerate(tickers):
        print(
            "Adding data for %s: %s out of %s" %
            (t[1], i+1, lentickers)
        )
        yf_data = get_daily_historic_data_google(t[1])
        print(yf_data)
        berak;
        #insert_daily_data_into_db('1', t[0], yf_data)
    print("Successfully added Yahoo Finance pricing data to DB.")

