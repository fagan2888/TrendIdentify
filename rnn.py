# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:17 2019

@author: 86156
"""

# build a RNN using keras 
# all data is from wind database

import numpy as np
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
import pandas as pd
#from WindPy import *
from config import *
import talib
from sqlalchemy import *
from ipdb import set_trace

#def loadDataFromTerminal(code, sdate, edate):   
#    w.start()
#    _,df = w.wsd(code, "dealnum,volume,amt,close", sdate, edate, usedf = True)
#    
#    return df

def indexMembers(indexCode):
    
    db = create_engine(uris['wind'])
    meta = MetaData(bind = db)
    t = Table('aindexmembers', meta, autoload = True)
    columns = [
            t.c.S_CON_WINDCODE.label('code'),
            ]
    sql = select(columns)
    sql = sql.where(t.c.S_INFO_WINDCODE == indexCode)
    sql = sql.where(t.c.S_CON_OUTDATE.is_(None))
    codes = pd.read_sql(sql, db)
    codes['code'] = codes['code'].apply(lambda x: x[0:6])
    codes = list(codes['code'])

    return codes


def loadData(scode, ecode, codes, sdate, edate):
    
    db = create_engine(uris['wind']) 
    meta = MetaData(bind = db)
    t = Table('asharedescription', meta,autoload = True)
    columns = [
            t.c.S_INFO_WINDCODE,
            ]
    sql = select(columns)
    if len(codes) != 0:
        sql = sql.where(t.c.S_INFO_CODE.in_(codes))
    else:
        sql = sql.where(t.c.S_INFO_CODE.between(scode, ecode))
    windCodes = list(pd.read_sql(sql, db)['S_INFO_WINDCODE'])
    
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
    sql = sql.where(t.c.S_INFO_WINDCODE.in_(windCodes))
    df1 = pd.read_sql(sql, db)
    df1.code = df1.code.apply(lambda x: x[0:6])
    df1.sort_values('date', ascending = True, inplace = True)
    df1 = df1.dropna(axis = 0, how = 'any')
    df1.set_index(['code','date'], inplace = True)
    
    t = Table('ashareeodprices', meta, autoload = True)
    columns = [
            t.c.S_INFO_WINDCODE.label('code'),
            t.c.TRADE_DT.label('date'),
            t.c.S_DQ_OPEN.label('open'),
            t.c.S_DQ_HIGH.label('high'),
            t.c.S_DQ_LOW.label('low'),
            t.c.S_DQ_CLOSE.label('close'),
            t.c.S_DQ_AMOUNT.label('amount'),
            t.c.S_DQ_VOLUME.label('volume'),
            t.c.S_DQ_PCTCHANGE.label('dailyReturn'),
            ]
    sql = select(columns)
    sql = sql.where(t.c.TRADE_DT.between(sdate, edate))
    sql = sql.where(t.c.S_INFO_WINDCODE.in_(windCodes))
    df2 = pd.read_sql(sql, db)
    df2.code = df2.code.apply(lambda x: x[0:6])
    df2.sort_values('date', ascending = True, inplace = True)
    df2 = df2.dropna(axis = 0, how = 'any')
    df2.set_index(['code','date'], inplace = True)

    df = pd.merge(df1, df2, left_index = True, right_index = True) 

    t = Table('asharemoneyflow', meta, autoload = True)
    columns = [
            t.c.S_INFO_WINDCODE.label('code'),
            t.c.TRADE_DT.label('date'),
            t.c.TRADES_COUNT.label('deals'),
            ]
    sql = select(columns)
    sql = sql.where(t.c.TRADE_DT.between(sdate, edate))
    sql = sql.where(t.c.S_INFO_WINDCODE.in_(windCodes))
    df3 = pd.read_sql(sql, db)
    df3.code = df3.code.apply(lambda x: x[0:6])
    df3.sort_values('date', ascending = True, inplace = True)
    df3 = df3.dropna(axis = 0, how = 'any')
    df3.set_index(['code','date'], inplace = True)

    df = pd.merge(df, df3, left_index = True, right_index = True)

    df['mbss'] = df.apply(lambda x: x['volume']/x['deals'], axis = 1)
    df = df.drop('deals',axis = 1)

    t = Table('ashareeodderivativeindicator', meta, autoload = True)
    columns = [
            t.c.S_INFO_WINDCODE.label('code'),
            t.c.TRADE_DT.label('date'),
            t.c.S_DQ_MV.label('marketPrice'),
            t.c.S_VAL_MV.label('totalPrice'),
            ]
    sql = select(columns)
    sql = sql.where(t.c.TRADE_DT.between(sdate, edate))
    sql = sql.where(t.c.S_INFO_WINDCODE.in_(windCodes))
    df4 = pd.read_sql(sql, db)
    df4.code = df4.code.apply(lambda x: x[0:6])
    df4.sort_values('date', ascending = True, inplace = True)
    df4 = df4.dropna(axis = 0, how = 'any')
    df4.set_index(['code','date'], inplace = True)

    df = pd.merge(df, df4, left_index = True, right_index = True)

    return df

def indicators(dfBasic):
    df = pd.DataFrame(columns = ['code','date','indicator1', 'indicator2', 'indicator3', 'indicator4', 'indicator5', 'indicator6','indicator7', 'indicator8','indicator9','dailyReturn']) 
#    MBSS = list()
#    for i in range(len(dfBasic)):
#        mbss = dfBasic['VOLUME'][i] / dfBasic['DEALNUM'][i]
#        MBSS.append(mbss)
#    macd,macdsignal,macdhist = talib.MACD(dfBasic['CLOSE'].values, fastperiod = 12, slowperiod = 26, signalperiod = 9)
#    ma5 = talib.MA(dfBasic['CLOSE'],timeperiod = 5, matype = 0)
#    ma30 = talib.MA(dfBasic['CLOSE'],timeperiod = 30, matype = 0)
#    ma180 = talib.MA(dfBasic['CLOSE'],timeperiod = 180, matype = 0)
#    ma300 = talib.MA(dfBasic['CLOSE'],timeperiod = 300, matype = 0)
#    indicator1 = MBSS
#    indicator2 = list(macd)
#    indicator3 = list(ma5)
#    indicator4 = list(ma30)
#    indicator5 = list(ma180)
#    indicator6 = list(ma300)

    indicator1 = list(dfBasic.activeBuy)
    indicator2 = list(dfBasic.activeSell)
    indicator3 = list(dfBasic.mainForceBuy)
    indicator4 = list(dfBasic.mainForceSell)
    indicator5 = list(dfBasic.amount)
    indicator6 = list(dfBasic.volume)
    indicator7 = list(dfBasic.mbss)
    indicator8 = list(dfBasic.marketPrice)
    indicator9 = list(dfBasic.totalPrice)
    df.code = list(dfBasic.index.get_level_values(0).values)
    df.date = list(dfBasic.index.get_level_values(1).values)
    df.indicator1 = indicator1
    df.indicator2 = indicator2
    df.indicator3 = indicator3
    df.indicator4 = indicator4
    df.indicator5 = indicator5
    df.indicator6 = indicator6
    df.indicator7 = indicator7
    df.indicator8 = indicator8
    df.indicator9 = indicator9
    df.dailyReturn = list(dfBasic.dailyReturn)
    
    df = df.dropna(axis = 0, how = 'any')
    df.sort_values(by = 'date', ascending = True, inplace = True)

    # make sure that every code has data every day. if not, clear all the data that day
    dfNew = pd.DataFrame(columns = df.columns)
    codeNumbers = len(set(list(df.code)))
    for date in list(df.date):
        if len(df[df.date == date]) == codeNumbers:
            dfNew = pd.concat([dfNew, df[df.date == date]], axis = 0)
    dfNew = dfNew.reset_index(drop = True)
    
    return dfNew

def transform(df, backSteps):
    timeLength = int(backSteps)
    members = len(set(list(df.code)))
    slides = int(len(df) - backSteps * members)
    parameters = df.shape[1] - 3
    x = df.copy().drop('dailyReturn',axis = 1)
    x.set_index('date', inplace = True)
    codes = list(set(list(x.code)))
    
    # normalization
    for i in range(1,x.shape[1]):
        minValue = min(x.iloc[:,i])
        maxValue = max(x.iloc[:,i])
        x.iloc[:,i] = x.iloc[:,i].map(lambda x: (x-minValue)/(maxValue-minValue))
    xg = x.groupby('code')
    
    # x should be a 3D tensor for RNN input
    X = np.zeros((slides, timeLength, parameters))
    y = list()
    for code in codes:
        x = xg.get_group(code).sort_index().drop('code', axis = 1)
        length = len(x)
        for j in range(length-timeLength):
            X[j,:,:] = x.iloc[j:j+timeLength,:]
        y = y + (list(df[df.code == code].dailyReturn)[backSteps:length])

    # if we do not care about the specific data or daily return and only care about the sign
    # change the y into 0,1 series in which 0 means minus and 1 means plus
    for i in range(len(y)):
        if y[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
 
    return X ,y

def divide(x, r):
    num = len(x)
    testNum = int(num*r/(1+r))
    trainNum = num - testNum
    train = x[0:trainNum]
    test = x[trainNum:num]

    return train, test

def build(timeLength, parameters):
    # define a LSTM model
    # input_shape(n_steps, n_parameters)
    model = Sequential()
    model.add(LSTM(200, input_shape=(timeLength, parameters)))
    model.add(Dense(100, activation = 'selu'))
    model.add(Dense(80, activation = 'selu'))
    model.add(Dense(20, activation = 'selu'))
    
#    model.add(Dense(1))
#    model.compile(optimizer = 'RMSprop', loss = 'mse', metrics = ['accuracy'])
    
    # if do not care about specific value but only care about the sign use following structure
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def train(x, y, model, e = 0.33, a = 0.90):
    # train the model
    for i in range(1,30000):
        model.fit(x, y, epochs = 1, batch_size = 16)
        [loss, acc] = model.evaluate(x, y)
        
#        if loss < e:
#            print('model well-trained! loss is less than:', loss)
#            return model
        
        # if do not care about specific value but only care about the sign use following structure
        if acc > a:
            print('model well-trained! loss is less than:', loss)
            return model
    
    print('model not well-trained! please change epoch and batch_size!')

    return model

def handle(scode = '000000', ecode = '999999', codes = [], sdate = '19900101', edate = '20200101', backSteps = 10):
    
    dfBasic = loadData(scode, ecode, codes, sdate, edate)
    df = indicators(dfBasic)
#    df.to_excel('basicData.xlsx')

#    df = pd.read_excel('basicData.xlsx', index_col = 0)
    df.code = df.code.apply(lambda x: str(x).zfill(6))
    df.date = df.date.map(lambda x: str(x))

    print('start from: ',min(df.date))
    print('end at: ', max(df.date))
    set_trace()
    #df = df[df['date'] > '20150101']
    #print(df)
    #set_trace()
    
    # transform dataframe into matrics that can feed into rnn
    # use the x values of the last backSteps to pridict y values now
    X, y = transform(df, backSteps) 
    
    # divided X and y into training group and testing group
    # the proportion of number of elements in testing group and training group is r
    r = 1 / 9 
    xTrain, xTest = divide(X, r)
    yTrain, yTest = divide(y, r)

    parameters = df.shape[1] - 3
    
    # x shape(slides, time_steps, parameters)
    # y shape(slides, returns)    
    proto = build(backSteps, parameters)

    proto.fit(xTrain,yTrain, epochs = 100, batch_size = 64)
    print('pre-train done!')
    set_trace()

    model = train(xTrain,yTrain,proto)
    [loss, acc] = model.evaluate(xTest, yTest)
    print('loss for test group is ', loss)
    print('accuracy for test group is', acc)
    #out = model.predict(xt, batch_size = 1)
    #print('predict value:',out)
    #if loss < (0.33 + 0.001):
    # if only care about sign use acc
    if acc > 0.8:
        model.save('new.h5')
        print('useful model saved!')
    else:
        print('model cannot be used for test group please change!')
    
    
if __name__ == '__main__':
    # use 沪深300 as default index
    #indexCode = '399300.SZ'
    #codes = indexMembers(indexCode)
    #codes = ['601318'] # 中国平安 排除
    #codes = ['000559'] # 万向钱潮
    #codes = ['002582'] # 好想你
    #codes = ['600276'] # 恒瑞医药 排除
    codes = ['000651'] # 格力电器 较好
    #codes = ['600519'] # 贵州茅台 排除
    output = handle(codes = codes, sdate = '19960701', edate = '20181230')
