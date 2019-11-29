# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:26:10 2019

@author: 86156
"""

# calculate Hurst index of a time series
# written by zhaoliyuan

import pandas as pd
import numpy as np
from sqlalchemy import *
from datetime import datetime
from ipdb import set_trace
from config import *

# load data from database
def loadData():
    db = create_engine(uris['base'])
    meta = MetaData(bind = db)
    t = Table('',meta, autoload = True)
    columns = [
            ,
            ,
            ]
    s = select(columns).where()
    df = pd.read_sql(s,db)
    df.sort_values(by = ,inplace = True,ascending = True)
    df.set_index([], inplace = True)
    
    return df

# calculate Hurst index
def calculate(df):
    
    return hurst