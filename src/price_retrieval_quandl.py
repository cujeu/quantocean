from datetime import datetime as dt
import warnings
import MySQLdb as mdb
import requests
import numpy
import pandas
import quandl
import time
#from passwords import PASSWORD, API_KEY

API_KEY = 'Jszm-d1BscjHYRVaMmUX'
#quandl_key = '-6X6UvP1aeit_zybGREM'
#API_KEY = 'v4WFg1TU4d4_qATreoLy'   # Jun's key
# Authenticate Session
quandl.ApiConfig.api_key = API_KEY

# Obtain a database connection to the MySQL instance
db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'password'
db_name = 'securities_master'
con = mdb.connect(db_host, db_user, db_pass, db_name)


"""
Obtains a list of the S&P500 ticker symbols from the database.
"""
def obtain_list_of_db_tickers():
    with con:
        cur = con.cursor()
        cur.execute("SELECT id, ticker FROM symbol")
        data = cur.fetchall()
        return [(d[0], d[1]) for d in data]


"""
Obtains Quandl data and returns a list of tuples.
"""
def get_daily_historic_data_quandl(ticker):
        #, start_date=(2008,1,1),
        #end_date=dt.date.today().timetuple()[0:3]):

    """ticker_tup = (ticker, start_date[0], start_date[1], start_date[2],
                  end_date[0], end_date[1], end_date[2], API_KEY)
    url = "https://www.quandl.com/api/v3/datasets/WIKI/"
    url += "%s.json?start_date=%s-%s-%s&end_date=%s-%s-%s&api_key=%s" % ticker_tup
    print(url)
    """
    prices = []
    ## quandl stop update after 2018/apr
    start_date = dt(year=2006, month=1, day=1)
    end_date = dt(year=2017, month=12, day=1) ##dt.today()

    # Try connecting to Quandl and obtaining the data
    # On failure, print an error message.
    try:
        """
        option 1:results as below
                ticker       date     ...       adj_close  adj_volume
        None                       ...                            
        0       MMM 2017-12-05     ...      238.260000   1543134.0
        1       MMM 2017-11-28     ...      235.630000   1797384.0
        start_date_str='2008-01-01'
        end_date_str='2018-3-01'
        data = quandl.get_table('WIKI/PRICES', ticker=ticker, date={'gte': start_date_str, 'lte': end_date_str})
        """

        """
        option 2:
        data = quandl.get("WIKI/{}".format(ticker.replace(".", "_")),
                        start_date=str(start_date[0]) + '-' + str(start_date[1]) + '-' + str(start_date[2]),
                        end_date=str(end_date[0]) + '-' + str(end_date[1]) + '-' + str(end_date[2]),
                        returns="numpy")
        for i in range(len(data)):
            date = data[i][0]
            Open = data[i][1]
            high = data[i][2]
            low = data[i][3]
            lose = data[i][4]
            volume = data[i][5]
            adj_open = data[i][8]
            adj_high = data[i][9]
            adj_low = data[i][10]
            adj_close = data[i][11]
            prices.append((date.date(), Open, high, low, close, volume,
                          adj_open, adj_high, adj_low, adj_close))
        """
        
        """
        option 3: results as belw
                    Open      High     ...       Adj. Close  Adj. Volume
        Date                             ...                              
        2008-01-02   84.23   84.7600     ...        64.029559    4453700.0
        2008-01-03   82.64   83.4800     ...        64.021817    2724100.0
        ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 
          'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']
        """

        quandl_code = "WIKI/" + ticker
        data = quandl.get(quandl_code, returns="pandas",
                         trim_start=start_date, trim_end=end_date,
                         authtoken=API_KEY)
        #data.to_csv('mmm.csv', sep='\t', encoding='utf-8')
        ## data type is dataframe
        ## print(list(data.columns.values))
        indexv = data.index.tolist()
        datav = data.values
        print('downloaded entries:',len(indexv))
        #print(indexv)
        #print (datav)
        ##for i in range(len(data[0])):
        ##    print(data[0][i])
        for i in range(len(indexv)):
            date = indexv[i].date()
            oprice = datav[i][7]
            high = datav[i][8]
            low = datav[i][9]
            close_price = datav[i][10]
            adj_close = datav[i][10]
            volume = datav[i][11]
            prices.append((date, oprice, high, low, close_price, adj_close, volume))
    except Exception as e:
        print("Could not download Quandl data: %s" % e)

    return prices


"""
Takes a list of tuples of daily data and adds it to the
MySQL database. Appends the vendor ID and symbol ID to the data.
daily_data: List of tuples of the OHLC data (with
adj_ohlc and volume)
"""
def insert_daily_data_into_db(data_vendor_id, symbol_id, daily_data):

    # Create the time now
    # now = dt.datetime.utcnow()

    # Amend the data to include the vendor ID and symbol ID
    daily_data = [(data_vendor_id, symbol_id, d[0], d[0], d[0],d[1], d[2], d[3], d[4], d[5], d[6]) for d in daily_data]
    #daily_data = [(data_vendor_id, symbol_id, d[0], now, now,
    #            d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9])
    #            for d in daily_data]

    # Create the insert strings
    column_str = "data_vendor_id, symbol_id, price_date, created_date, last_updated_date, open_price, high_price, low_price, close_price, adj_close_price, volume"
    ##, adj_open_price, adj_high_price, adj_low_price, adj_close_price"
    insert_str = ("%s, " * 11)[:-2]
    final_str = "INSERT INTO daily_price (%s) VALUES (%s)" % (column_str, insert_str)
    # print(final_str)

    # Using the MySQL connection, carry out an INSERT INTO for every symbol
    with con:
        cur = con.cursor()
        cur.executemany(final_str, daily_data)


if __name__ == "__main__":

    # This ignores the warnings regarding Data Truncation
    warnings.filterwarnings('ignore')

    # Loop over the tickers and insert the daily historical data into the database
    tickers = obtain_list_of_db_tickers()
    lentickers = len(tickers)
    for i, t in enumerate(tickers):
        print("Adding data for %s: %s out of %s" %
                (t[1], i+1, lentickers))
        if i > 250:
            API_KEY = 'v4WFg1TU4d4_qATreoLy'   # Jun's key

        quandl_data = get_daily_historic_data_quandl(t[1])
        ## quandl_data row like datetime.date(2018, 3, 7), 231.22, 236.22, 230.59, 235.57, 235.57, 2213792.0
        ## print(quandl_data)
        insert_daily_data_into_db('1', t[0], quandl_data)
        # Wait for 500 milliseconds, quandl allonw 10 accesses per second
        time.sleep(.500)
    print("Successfully added Quandl pricing data to DB.")

