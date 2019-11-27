
# ARIMA algorithm
# written by zhaoliyuan

import numpy as np
import pandas as pd
import matplotlib as plt
from sqlalchemy import *
from optparse import OptionParser
import config

class ARIMA(code):
    
    def __init__(self,code):
        self.code = code

    def loadData(self):
        code = self.code
        
        db = creat_engine(uris['wind'])
        meta = MetaData(bind = db)
        t = Table('ashareeodprices', meta, autoload = True)
        columns = [
                t.c.S_INFO_WINDCODE,
                t.c.TRADE_DT,
                t.c.S_DQ_ADJOPEN,
                t.c.S_DQ_ADJHIGH,
                t.c.S_DQ_ADJLOW,
                t.c.S_DQ_ADJCLOSE,
                t.c.S_DQ_CHANGE,
                t.c.S_DQ_PCTCHANGE,
                ]
        s = select(columns).where(t.c.S_INFO_WINDCODE[0:6] == code)
        df = pd.read_sql(s, db)

        return df

    def findD():
        
        

    def calParams(dfTrain):

        params = [p,q]
        return params

    def evaluate(dfTest, ):


if __name__ == '__main__':
    
    defaultCode = '601318'
    parse = OptionParser()
    parse.add_option('-c', '--code', help = 'use code to choose', default = defaultCode)
    options, args = opt.parse()
    ARIMA = ARIMA(code)
    df = ARIMA.loadData()
    dfTrain = df.iloc[]
    dfTest = df.iloc[]
    params = ARIMA.calParams(dfTrain)
    ARIMA.evaluate(dfTest, )
    