import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
#最小二乘
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
#获得汇总信息
#import pandas_datareader as pdr

#from lecture_code_03 import price_retrieval as pr
#from src import price_db as prdb
import price_db.py as prdb


def plot_price_series(df, ts1, ts2, pm_start, pm_end):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(pm_start, pm_end)
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()

def plot_residuals(df, pm_start, pm_end):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(pm_start, pm_end)
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()

def plot_adf(TICKER_A, TICKER_B, pm_start, pm_end):

    df_a = prdb.retrieve_daily_price(TICKER_A, columns=('price_date', 'adj_close_price'), start=pm_start, end=pm_end)
    df_b = prdb.retrieve_daily_price(TICKER_B, columns=('price_date', 'adj_close_price'), start=pm_start, end=pm_end)

    df = pd.DataFrame(columns=(TICKER_A, TICKER_B))
    df[TICKER_A] = df_a["adj_close_price"]
    df[TICKER_B] = df_b["adj_close_price"]
    # print(df)

    # Plot the two time series
    plot_price_series(df, TICKER_A, TICKER_B, pm_start, pm_end)

    # Display a scatter plot of the two time series
    plot_scatter_series(df, TICKER_A, TICKER_B)

    # Calculate optimal hedge ratio "beta"
    # Step 1: regress on variable on the other
    res = sm.OLS(endog=df[TICKER_B], exog=df[TICKER_A]).fit()
    st, data, ss2 = summary_table(res, alpha=0.05)   # 置信水平alpha=5%，st数据汇总，data数据详情，ss2数据列名

    beta_hr = data[:, 2]  # 等价于res.fittedvalues
    # beta_hr = res.fittedvalues

    # Calculate the residuals of the linear combination
    # Step 2: obtain the residual (ols_resuld.resid)
    df["res"] = df[TICKER_A] - beta_hr*df[TICKER_B]

    # Plot the residuals
    plot_residuals(df, pm_start, pm_end)

    # Calculate and output the CADF test on the residuals
    # Step 3: apply Augmented Dickey-Fuller test to see whether
    #        the residual is unit root
    #cadf = ts.adfuller(res.resid)
    cadf = ts.adfuller(df["res"])
    pprint.pprint(cadf)
    """
    (3.3856214189376854,  ADF Statistic
     1.0,               p-value
     9,                 usedlag 
     2278,              nobs  
     {'1%': -3.433223873595976,     Critical Values
      '10%': -2.5674458705439904,
      '5%': -2.862809607710201},
     29774.394211738363    icbest
     )
    """
    
    rcADF = False
    if cadf[0] > cadf[4].get("5%"):
        print("because %s > %s,  %s and %s prices are random walk" % (cadf[0], cadf[4].get("5%"), TICKER_A, TICKER_B))
    else:
        rcADF = True
        print("because %s < %s,  %s and %s prices are NOT random walk" % (cadf[0], cadf[4].get("5%"), TICKER_A, TICKER_B))
    return rcADF

def compare_adf(TICKER_A, TICKER_B, pm_start, pm_end):

    rcADF = False
    df_a = prdb.retrieve_daily_price(TICKER_A, columns=('price_date', 'adj_close_price'), start=pm_start, end=pm_end)
    df_b = prdb.retrieve_daily_price(TICKER_B, columns=('price_date', 'adj_close_price'), start=pm_start, end=pm_end)
    
    # print(df_a.shape[0],df_b.shape[0]) 
    #TODO: size alignment for the two df
    if (df_a.shape[0] < 2000):
        return rcADF 
    if (df_a.shape[0] != df_b.shape[0]) :
        return rcADF 

    df = pd.DataFrame(columns=(TICKER_A, TICKER_B))
    df[TICKER_A] = df_a["adj_close_price"]
    df[TICKER_B] = df_b["adj_close_price"]
    # print(df)

    # Calculate optimal hedge ratio "beta"
    # Step 1: regress on variable on the other
    res = sm.OLS(endog=df[TICKER_B], exog=df[TICKER_A]).fit()
    # 置信水平alpha=5%，st数据汇总，data数据详情，ss2数据列名
    st, st_data, ss2 = summary_table(res, alpha=0.05)

    beta_hr = st_data[:, 2]  # 等价于res.fittedvalues
    # beta_hr = res.fittedvalues

    # Calculate the residuals of the linear combination
    # Step 2: obtain the residual (ols_resuld.resid)
    df["res"] = df[TICKER_A] - beta_hr*df[TICKER_B]

    # Calculate and output the CADF test on the residuals
    # Step 3: apply Augmented Dickey-Fuller test to see whether
    #        the residual is unit root
    #cadf = ts.adfuller(res.resid)
    cadf = ts.adfuller(df["res"])
    # pprint.pprint(cadf)
    
    if cadf[0] < cadf[4].get("5%"):
        rcADF = True
    return rcADF

if __name__ == "__main__":
    start_date = datetime.datetime(2008, 4, 1)
    end_date = datetime.datetime(2017, 5, 1)
    start_a = 1
    if len(sys.argv) == 2:  #one parameter
        start_a = int(sys.argv[1])

    ## rc = plot_adf("AMZN", "AAPL", start_date, end_date)
    sym_df = prdb.retrieve_id_ticker()
    idx_a = 0
    idx_b = 0
    ticker_a = ""
    ticker_b = ""
    skip_list = ["AIZ", "ETFC"]
    for index_a, row_a in sym_df.iterrows():
        idx_a = index_a 
        ticker_a = row_a["ticker"]
        if idx_a < start_a:
            continue

        for index_b, row_b in sym_df.iterrows():
            idx_b = index_b
            ticker_b = row_b["ticker"]
            if (idx_b > idx_a) and (not (ticker_a in skip_list)) and (not (ticker_b in skip_list)):
                print("compare", idx_a, idx_b, ticker_a, ticker_b)
                try:
                    rc = False
                    rc = compare_adf(ticker_a, ticker_b, start_date, end_date)
                    if rc == True:
                        print(ticker_a, ticker_b, "NON-random walking")
                except:
                    print("ADF Test Error")

        # break
    print("ADF test Done")

