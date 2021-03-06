# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:15:11 2014

@author: colin4567
"""

class Event(object):
    """
    Base class providing an interface for all subsequent (inherited)
    events, that will trigger further events in the trading infrastructure
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update
    with corresponding bars
    """
    def __init__(self):
        self.type = 'MARKET'
        
class FinishEvent(Event):
    """
    stop backtest
    """
    def __init__(self):
        self.type = 'FIN'
        

class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon
    """
    def __init__(self, strategy_id, symbol, datetime, signal_type, strength):
        """
        Initializes the SignalEvent.
        
        Parameters:
        strategy_id - Unique identifier for strategy that generated the signal
        symbol - ticker symbol
        datetime - timestamp of when signal was generated
        signal_type - 'LONG' or 'SHORT' or 'EXIT'
        strength - Adjustment factor "suggestion" used to scale
                    quantity at the portfolio level. Used for mean reversion
        """
        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength

class TimerEvent(Event):
    """
    Handles the event of set a sell timer a Strategy object.
    """
    def __init__(self, strategy_id, symbol, datetime, signal_type, strength, window):
        """
        Initializes the SignalEvent.
        
        Parameters:
        strategy_id - Unique identifier for strategy that generated the signal
        symbol - ticker symbol
        datetime - timestamp of when signal was generated
        window  
        signal_type - 'LONG' or 'SHORT' or 'EXIT'
        strength - Adjustment factor "suggestion" used to scale
                    quantity at the portfolio level. Used for mean reversion
        """
        self.type = 'TIMEOUT'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.window = window
        self.strength = strength


class OrderEvent(Event):
    """
    Handles the event of sending on Order to the execution system
    
    Parameters:
    symbol - instrument to trade
    order_type - 'MKT' or 'LMT' for Market and Limit
    quantity - Non-negative integer 
    direction - 'BUY' or 'SELL'
    """
    def __init__(self, symbol, order_type, quantity, direction, position_change):
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.position_change = position_change
    
    def print_order(self):
        """
        Outputs the values within the order
        """
        print ("Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s, Position_Change=%s" % \
                (self.symbol, self.type, self.quantity, self.direction, self.position_change))
                

class FillEvent(Event):
    """
    Encapsulation of a filled order
    If commision is not provided, the fill object will calculate it 
    based on quantity and broker fees
    
    Parameters:
    timeindex - bar resolution when the order was filled
    symbol - instrument filled
    exchange - where the order was filled
    quantity - filled quantity
    direction - 'BUY', 'SELL'
    fill_cost - holdings value in dollars
    commission - optional
    """
    def __init__(self, timeindex, symbol, exchange, quantity, direction, position_change, commission=None):
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.position_change = position_change
        
        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission
    
    def calculate_commission(self):
        """This is the rate for Schwab online execution"""
        execution_fee = 4.95
        return execution_fee
    
