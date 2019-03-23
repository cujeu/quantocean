# -*- coding: utf-8 -*-
"""
Created on Dec/18 QDA backtest
ref:https://github.com/jenniyanjie/successful-algorithmic-trading/blob/a34bafefebe1afb96836e6789dda34b4564b0e19/chapter15/snp_forecast.py
@author: Jun Chen
"""

from __future__ import print_function

import datetime
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA

#import from same directory
from config import Config
from backtest import Backtest
from strategy import Strategy
from event import SignalEvent
from event import TimerEvent
from event import FinishEvent
from data import MySQLDataHandler
from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
#from create_lagged_series import create_lagged_series

import sys
sys.path.append('/home/jun/proj/quantocean/src')
from make_data import *

# Add the folder path to the sys.path list

class QDAForecastStrategy(Strategy):
    """
    Default window is 34/144
    """
    def __init__(self, conf, bars, events, timerEvents, short_window=5, long_window=14):
        """
        S&P500 forecast strategy. It uses a Quadratic Discriminant
        Analyser to predict the returns for a subsequent time
        period and then generated long/exit signals based on the
        prediction.
        
        Initializes the buy and hold strategy
        Parameters:
        bars - DataHandler object that provides bar info
        events - Event queue object
        short/long windows - moving average lookbacks
        """
        self.conf = conf
        self.logger = self.conf.logger
        self.logger.info('QDAForecastStrategy __init__')
        self.bars = bars    #bars type is 'data.MySQLDataHandler'
        self.symbol_list = self.conf.symbol_list
        self.events = events
        self.timerEvents = timerEvents
        self.datetime_now = datetime.datetime.utcnow()
        self.short_window = short_window
        self.long_window = long_window
        

        ##self.model_start_date = datetime.datetime(2010,1,10)
        ##self.model_end_date = datetime.datetime(2015,12,31)
        ##self.model_start_test_date = datetime.datetime(2015,1,1)
        self.model_start_date = self.conf.start_date
        self.model_train_date = self.conf.train_date
        self.model_end_date = self.conf.end_date
        self.model_test_date = self.conf.test_date

        self.long_market = False
        self.short_market = False
        self.bar_index = 0

        self.pred = []
        self.model = self.create_symbol_forecast_model()
        #set to true if a symbol if in the market
        self.bought = self._calculate_initial_bought()

    
    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols and sets them to 'OUT'
        """
        bought = {}        
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought
    
    def create_symbol_forecast_model(self):
        #start_test = self.model_start_test_date
        # Create a lagged series of the S&P500 US stock market index
        # snpret = create_lagged_series(
        #    self.symbol_list[0], self.model_start_date, 
        #    self.model_end_date, lags=5 )

        # Use the prior two days of returns as predictor 
        # values, with direction as the response
        #X = snpret[["Lag1","Lag2"]]
        #y = snpret["Direction"]

        # Create training and test sets
        #X_train = X[X.index < start_test]
        #X_test = X[X.index >= start_test]
        #y_train = y[y.index < start_test]
        #y_test = y[y.index >= start_test]
       
        snpret = make_dataset1( self.symbol_list[0], 'adj_close_price',
                                self.model_start_date, self.model_end_date)
        X_train, X_test, y_train, y_test = make_train_3way(snpret, 'adj_close_price',
                            5, 5.5, self.model_train_date, self.model_test_date)
        """ 
        y_train(1240,) y_test(252,) are  'pandas.core.series.Series' , y_test is the true result
        price_date
        2010-01-29    0.0
        ...
        2014-12-30    0.0
        2014-12-31    0.0
        X_train(1240, 13) and X_test(252, 13) is 'pandas.core.frame.DataFrame'
                    open_price  high_price  low_price      ...       williams_r15    RSI_15     pvt_ema15
        price_date                                         ...                                           
        2015-01-02    105.8210    105.8685   101.9829      ...           0.642600  0.261353  4.355163e+07
        2015-01-05    102.8760    103.2180   100.1399      ...           1.001175  0.216901  4.336671e+07
        ...
        2015-12-30    104.9044    105.0204   103.5518      ...           0.865691  0.371887  3.559120e+07
        2015-12-31    103.3876    103.4069   101.2717      ...           1.025577  0.282320  3.539544e+07
        """
        
        model = QDA()
        model.fit(X_train, y_train)
        cat_ser = pd.concat([y_train, y_test])
        self.pred = model.predict(X_test)
        #pred(252,) is 'numpy.ndarray' , [0. -1. -1.  0.  0 ..... 0.  0.  0.  0.  0.]
        #merge pred to self.symbol_data[s] or another dataFrame with datatime
        p_index = self.pred.shape[0]
        idx = -1
        for x in range(p_index):
            cat_ser.iloc[idx] = self.pred[p_index+idx]
            idx -= 1

        self.strategy_df = cat_ser.to_frame()
        return model

    def calculate_signals(self, event):
        """
        calculate_signals is the implementation of abstract method
        Generates a new SignalEvent object and places it on the
        Event queue, and updates the bought attribute.
        
        Parameters:
        event - a MarketEvent object
        
        To Do:
        strength needs to be applied to all positions when a new symbol comes on, not just to new position
        """
        if event.type == 'MARKET':
            for symbol in self.bars.latest_symbol_list: 
                bars = self.bars.get_latest_bars_values(symbol, "adj_close", N=self.long_window)
                
                if bars is not None and bars != []:
                    dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    #this is where you set percentage of capital - 
                    #how can I easily rescale previous positions when new symbol becomes eligable
                    strength = 0.25 / len(self.bars.latest_symbol_list)
                    strategy_id = 2
                    self.bar_index += 1
                    #dt.type is Timestamp, datetime
                    if (dt >= self.model_end_date):
                        print('hit and gen the end')
                        self.events.put(FinishEvent())
                    elif (dt > self.model_test_date):
                        pred_dir = self.strategy_df.loc[dt.date()][0]
                        if pred_dir > 0: ## and not self.long_market:
                            sig_dir = 'LONG'
                            self.long_market = True
                            signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                            self.events.put(signal)
                            self.bought[symbol] = 'LONG'    ## change state
                            # QDA strategy is buy and sell in 5 days, so 
                            # put another timer event
                            sig_dir = 'EXIT'
                            signal = TimerEvent(strategy_id, symbol, dt, sig_dir, strength, 5)
                            self.timerEvents.put(signal)
                        """
                        ## do not short for a while
                        elif pred_dir < 0 and self.long_market:
                            sig_dir = 'SHORT'
                            self.long_market = False
                            signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                            self.events.put(signal)
                            self.bought[symbol] = 'OUT'  ## change state
                        """


if __name__ == '__main__':
    
    import os
    
    ##symbol_list = ['ba', 'noc', 'lmt', 'nwsa', 'goog', 'alle', 'navi'] 
    ##symbol_list = ['AAPL']
    #broken = ['nwsa', 'goog', 'alle', 'navi']
    #symbols =  pd.read_csv("C:/Users/colin4567/Dropbox/EventTradingEngine/getData/testData/sp500ticks.csv"); symbol_list = symbols['tickers'].tolist()
    ##initial_capital = 1000000.0 #1m
    ##start_date = datetime.datetime(2010,1,1,0,0,0)
    ##end_date = datetime.datetime(2015,12,31,0,0,0)    ##start <5 years> end
    ##test_date = datetime.datetime(2015,1,1,0,0,0)   ##test_last year
    ##heartbeat = 0.0
    ##data_feed = 2 # 1 is csv, 2 is MySQL
    conf = Config()

    for asymbol in conf.whole_list:
        conf.symbol_list = []
        conf.symbol_list.append(asymbol)
        if conf.data_feed == 1:
            if os.path.isdir(conf.csv_dir):
                csv_dir = os.path.normpath(conf.csv_dir)
            else:
                raise SystemExit("No csv dir found ")
            
            backtest = Backtest(conf,
                                HistoricCSVDataHandler, 
                                SimulatedExecutionHandler, 
                                Portfolio, 
                                QDAForecastStrategy)
            backtest.simulate_trading() ## trigger the backtest
                        
        elif conf.data_feed == 2:
        
            backtest = Backtest(conf,
                                MySQLDataHandler, 
                                SimulatedExecutionHandler, 
                                Portfolio, 
                                QDAForecastStrategy)
            backtest.simulate_trading() ## trigger the backtest
    #end of for whole_list            
                    
                    
                 
