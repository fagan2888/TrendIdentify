# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:17 2019

@author: 86156
"""

# build a RNN using tensorflow 

import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import pandas as pd
from WindPy import *
from config import *
import talib
from sqlalchemy import *

def loadDataFromTerminal(code, sdate, edate):   
    w.start()
    _,df = w.wsd(code, "dealnum,volume,amt,close", sdate, edate, usedf = True)
    
    return df

def loadData(scode = '000000', ecode = '999999', sdate = '19900101', edate = '20200101'):
    db = create_engine(uris['wind_db']) 
    meta = MetaData(bind = db)
    t = Table('asharel2indicators', meta, autoload = True)
    columns = [
            t.c.S_INFO_WINDCODE.label('code'),
            t.c.TRADE_DT.label('date'),
            t.c.S_LI_INITIATIVEBUYRATE.label('activeBuy'),
            t.c.S_LI_INITIATIVESELLRATE.label('activeSell'),
            t.c.S_LI_LARGEBUYRATE.label('mainForceBuy'),
            t.c.S_LI_LARGESELLRATE.label('mainForceSell'),
            ]
    sql = select(columns)
    sql = sql.where(t.c.TRADE_DT.between(sdate, edate))
    df = pd.read_sql(sql, db)
    df.code = df.code.apply(lambda x: x[0:6])
    df = df[df['code'] < ecode && df['code'] > scode]
    df.sort_values('date', ascending = True, inplace = True)
    df = df.dropna(axis = 0, how = 'any')
    df.set_index(['date','code'] inplace = True)

    return df

def indicators(code, sdate, edate):
    df = pd.DataFrame(columns = ['code','date','indicator1', 'indicator2', 'indicator3', 'indicator4', 'indicator5', 'indicator6', 'close']) 
    dfBasic = loadData(code, sdate, edate).dropna(how = 'any')
    MBSS = list()
    for i in range(len(dfBasic)):
        mbss = dfBasic['VOLUME'][i] / dfBasic['DEALNUM'][i]
        MBSS.append(mbss)
    macd,macdsignal,macdhist = talib.MACD(dfBasic['CLOSE'].values, fastperiod = 12, slowperiod = 26, signalperiod = 9)
    ma5 = talib.MA(dfBasic['CLOSE'],timeperiod = 5, matype = 0)
    ma30 = talib.MA(dfBasic['CLOSE'],timeperiod = 30, matype = 0)
    ma180 = talib.MA(dfBasic['CLOSE'],timeperiod = 180, matype = 0)
    ma300 = talib.MA(dfBasic['CLOSE'],timeperiod = 300, matype = 0)
    indicator1 = MBSS
    indicator2 = list(macd)
    indicator3 = list(ma5)
    indicator4 = list(ma30)
    indicator5 = list(ma180)
    indicator6 = list(ma300)
    
    df.code = code
    df.date = dfBasic.index
    df.indicator1 = indicator1
    df.indicator2 = indicator2
    df.indicator3 = indicator3
    df.indicator4 = indicator4
    df.indicator5 = indicator5
    df.indicator6 = indicator6
    df.close = dfBasic.CLOSE
    
    df = df.dropna(axis = 0, how = 'any')
    df.sort_values(by = 'date', ascending = True, inplace = True)
    df.set_index('date', inplace = True)
    
    return df

def build():
    # define a LSTM model
    # input_shape(n_steps, n_parameters)
    model = Sequential()
    model.add(LSTM(100, input_shape=(1,6)))
    model.add(Dense(80, activation = 'selu'))
    model.add(Dense(40, activation = 'selu'))
    model.add(Dense(10, activation = 'selu'))
    model.add(Dense(1))
    
    model.compile(optimizer = 'RMSprop', loss = 'mse', metrics = ['accuracy'])
    
    return model

def train(x, y, model, r = 0.33):
    # train the model
    for i in range(1,30000):
        model.fit(x, y, epochs = 1, batch_size = 16)
        [loss, acc] = model.evaluate(x, y)
        if loss < r:
            break
        
    return model

def handle(code, sdate, edate):
    
    df = indicators(code, sdate, edate)
    x = df.copy().drop('close',axis = 1)
    x = x.drop('code',axis = 1)
    # x should be a 3D tensor for RNN input
    X = np.zeros((201,1,6))
    X[:,0,:] = x
    y = df.close
    # x shape(slides,n_steps, n_parameters)
    # y shape(slides,1)    
    
    proto = build()
    model = train(X,y,proto)
    [loss, acc] = model.evaluate(xt, yt)
    print(loss)
    print(acc)
    out = model.predict(xt, batch_size = 1)
    print(out)
    model.save('new.h5')
    
    
if __name__ == '__main__':
    
    df = loadData('000001', '000002')
    print(df)
    set_trace()
    
    output = handle('000001.SZ','2019-01-02','2019-12-05')
