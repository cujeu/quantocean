# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:27:00 2014

@author: colin4567
"""

import datetime
import pprint
import queue
import time
import logging
import logging.config
from config import Config
from event import SignalEvent
from event import TimerEvent

class Backtest(object):
    """
    Encapsulates the settings and components for carrying out an event-driven
    backtest.
    """
    def __init__(self,conf, data_handler, execution_handler, portfolio, strategy):
        """
        Initializes the backtest.
        
        Parameters:
        csv_dir - the hard root to the CSV data directory
        symbol_list - list of symbol string
        initial_capital - starting capital
        heartbeat - backtest heartbeat in seconds
        start_date - starting date of strategy
        data_handler - (Class) handles the market data feed
        execution_handler - (Class) handles the orders/fills for trades
        portfolio - (Class) Keeps track of current and prior positions
        strategy - (Class) generates signals based on market data
        """
        self.conf = conf
        self.logger = self.conf.logger
        self.logger.info('backtest __init__')
        ##self.symbol_list = conf.symbol_list
        ##self.data_feed = conf.data_feed
        ##self.initial_capital = conf.initial_capital
        ##self.heartbeat = conf.heartbeat
        ##self.start_date = conf.start_date
        ##self.end_date = conf.end_date
        ##self.test_date = conf.test_date
        ##self.csv_dir = conf.csv_dir
        ##self.db_host = conf.db_host
        ##self.db_user = conf.db_user
        ##self.db_pass = conf.db_pass
        ##self.db_name = conf.db_name

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        
        self.events = queue.Queue()
        self.timerEvents = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        
        self._generate_trading_instances(conf.data_feed)
    

    def _generate_trading_instances(self, data_feed):
        """
        Generates the trading instance objects from their class types
        """
        print ("Creating DataHandler, Strategy, Portfolio, and ExecutionHandler...")
        
        if self.conf.data_feed == 1: #HistoricCSVDataHandler
            self.data_handler = self.data_handler_cls(self.conf, self.events)
        elif self.conf.data_feed == 2: #MySQLDataHandler
            self.data_handler = self.data_handler_cls(self.conf, self.events)
         
        # create strtegy class 
        self.strategy = self.strategy_cls(self.conf,self.data_handler, self.events, self.timerEvents)
        
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, 
                                            self.conf.start_date, self.conf.initial_capital)
                                            
        self.execution_handler = self.execution_handler_cls(self.events)
        
    
    def _run_backtest(self):
        """
        Executes the backtest. Outer loop keeps track of the heartbeat, 
        inner checks if there is an event in the Queue object and sacts on it 
        by calling the appropriate method on the necessary object.
        """
        i = 0
        running = True
        while running:
            i += 1
            #Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
                self.data_handler.update_symbol_list(min_bars=self.strategy.long_window)
            else:
                break
            
            qs = self.timerEvents.qsize()
            #handle the timer events, process untill empty queue
            while (qs > 0):
                try:
                    timerEvent = self.timerEvents.get(False) #block=False
                    qs -= 1
                except queue.Empty:  ##self.timerEvents.Empty:
                    break
                else:
                    if timerEvent is not None:
                        if timerEvent.window >= 1:
                            signal = TimerEvent(timerEvent.strategy_id, timerEvent.symbol, timerEvent.datetime, timerEvent.signal_type, timerEvent.strength, timerEvent.window-1)
                            self.timerEvents.put(signal)
                        else:
                            signal = SignalEvent(timerEvent.strategy_id, timerEvent.symbol, timerEvent.datetime, timerEvent.signal_type, timerEvent.strength)
                            self.events.put(signal)
            #end of timer event 

            #handle the events, process untill empty queue
            while True:
                try:
                    event = self.events.get(False) #block=False
                except queue.Empty: #self.events.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)
                        
                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)
                    
                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                        
                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

                        elif event.type == 'FIN':
                            print('End backtest')
                            running = False
                            break
            #end of event 
                            
            time.sleep(self.conf.heartbeat)
    
    
    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """

        self.portfolio.create_equity_curve_dataframe()
        self.portfolio.create_positioning_dataframe()
        
        print( "Creating summary stats...")
        stats = self.portfolio.output_summary_stats()
        
        print(  "Creating the equity curve...\n")
        print(  self.portfolio.equity_curve[:50])
        print('')        

        print(  "Ending equity curve...\n")
        print(  self.portfolio.equity_curve.tail(10)) 
        
        
        print(  "Creating the historical positioning...\n")
        print(  self.portfolio.positions.tail(10))
        print('')
        pprint.pprint(stats)
        
        print('')
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)
        
        
    
    def simulate_trading(self):
        """
        This is called from __main__.
        Simulates the backtest and outputs portfolio performance
        """
        #self.logger.info('backtest ------ %s',self.conf.symbol_list[0])
        self._run_backtest()
        self._output_performance()
        
        
