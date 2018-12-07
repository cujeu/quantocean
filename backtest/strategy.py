# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:35:42 2014

@author: colin4567
"""

import numpy as np
import pandas as pd
import datetime
import queue
pd.set_option('notebook_repr_html', False)

from abc import ABCMeta, abstractmethod

from event import MarketEvent

class Strategy(object):
    """
    Abstract base class providing an interface for all strategy objects.
    
    Goal: derived strategy object generates Signal objects for particular
        symbols based on the inputs of Bars generated by DataHandler.
    
    Obtains the Bar tuples from a queue object - designed to work with historic
    and live data
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def calculate_signals(self):
        """
        Provides the mechanisms to calculate the list of signals
        """
        raise NotImplementedError("Should implement calculate_signals()")
