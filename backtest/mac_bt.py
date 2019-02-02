# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:50:53 2014

@author: colin4567
"""

import numpy as np
import pandas as pd
import datetime

from config import Config
from backtest import Backtest
from data import HistoricCSVDataHandler
from data import MySQLDataHandler
from event import FinishEvent
from event import SignalEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from strategy import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    Default window is 34/144
    """
    def __init__(self, conf,bars, events, timerEvents, short_window=34, long_window=144):
        """
        Initializes the buy and hold strategy
        
        Parameters:
        bars - DataHandler object that provides bar info
        events - Event queue object
        short/long windows - moving average lookbacks
        """
        self.conf = conf
        self.bars = bars
        self.symbol_list = self.conf.symbol_list
        self.events = events
        self.timerEvents = timerEvents
        self.short_window = short_window
        self.long_window = long_window
        
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
                """
                bars is 'numpy.ndarray', look like 
                [18.0241 17.9855 17.6026 17.4895 17.5229 17.4239 17.4625 17.9251 17.654
                    ......
                 25.4753 25.7491 25.9727 26.8646 27.1948 26.8723 27.1987 27.082  27.5508]
                """ 
                # dt start from 2010-01-05 , which is conf.start_date
                dt = self.bars.get_latest_bar_datetime(symbol)
                if (dt >= self.conf.end_date):
                    self.events.put(FinishEvent())
                
                elif bars is not None and bars != []:
                    #bars end at 'current date', now calculate the SMA
                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])
                    sig_dir = ""
                    strength = 1.0 / len(self.bars.latest_symbol_list) #this is where you set percentage of capital - how can I easily rescale previous positions when new symbol becomes eligable
                    strategy_id = 1
                    
                    if short_sma > long_sma and self.bought[symbol] == 'OUT':
                        #short_sma > long_sma is LONG trigger
                        sig_dir = 'LONG'
                        #SignalEvent set event.type to 'SIGNAL', then 
                        #backtest process it
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'    ## change state
                    
                    elif short_sma < long_sma and self.bought[symbol] == 'LONG':
                        #short_sma < long_sma is EXIT trigger
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'  ## change state


if __name__ == '__main__':
    
    import os
    
    ##symbol_list = ['ba', 'noc', 'lmt', 'nwsa', 'goog', 'alle', 'navi'] 
    #symbol_list = ['AAPL']
    #broken = ['nwsa', 'goog', 'alle', 'navi']
    #symbols =  pd.read_csv("C:/Users/colin4567/Dropbox/EventTradingEngine/getData/testData/sp500ticks.csv"); symbol_list = symbols['tickers'].tolist()
    #initial_capital = 1000000.0 #1m
    #start_date = datetime.datetime(2010,1,1,0,0,0)
    #end_date = datetime.datetime(2015,12,31,0,0,0)    ##start <5 years> end
    #test_date = datetime.datetime(2015,1,1,0,0,0)   ##test_last year
    #heartbeat = 0.0
    #data_feed = 2 # 1 is csv, 2 is MySQL

    conf = Config()
    
    if conf.data_feed == 1:
        if os.path.isdir("/home/jun/proj/quantocean/eventDriven/testData"):
            csv_dir = os.path.normpath("/home/jun/proj/quantocean/eventDriven/testData")
        else:
            raise SystemExit("No csv dir found ")
            
        backtest = Backtest(conf,
                            HistoricCSVDataHandler, 
                            SimulatedExecutionHandler, 
                            Portfolio, 
                            MovingAverageCrossStrategy)
                        
    elif conf.data_feed == 2:
        
        backtest = Backtest(conf,
                            MySQLDataHandler, 
                            SimulatedExecutionHandler, 
                            Portfolio, 
                            MovingAverageCrossStrategy)
                  
    
    backtest.simulate_trading() ## trigger the backtest
                    
                    
                    
                    
                    
                    
