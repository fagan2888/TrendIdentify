
# ARIMA algorithm
# written by zhaoliyuan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import *
from optparse import OptionParser
from config import *
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import itertools
import seaborn as sns

class ARI(code):
    
    def __init__(self,code):
        self.code = code

    def loadData(self):
        code = self.code
        
        db = create_engine(uris['wind'])
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
        s = select(columns).where(t.c.S_INFO_WINDCODE.like(str(code)+'%'))
        df = pd.read_sql(s, db)
        df.sort_values(by = 'TRADE_DT', ascending = True, inplace = True)
        df.set_index('TRADE_DT', inplace = True)

        return df

    def findD(dfTrain,col):
        plt.plot(dfTrain['TRADE_DT'], dfTrain[col])
        diff1 = dfTrain[col].diff(1)
        plt.plot(dfTrain['TRADE_DT'], diff1)
        diff2 = diff1.diff(1)
        plt.plot(dfTrain['TRADE_DT'], diff2)
        

    def calParams(dfTrain,col, pMin,pMax, qMin, qMax):
#        std = dfTrain[col].std()
#        acf = stattools.acf(dfTrain[col], nlags = 100)
#        pacf = stattools.pacf(dfTrain[col], nlags = 100)
#        plot_acf(acf)
#        plot_pacf(pacf)
        d = 2
    
# way no.1
        BICResult = pd.DataFrame(
                index = ['AR{}'.format(i) for i in range(pMin, pMax+1)],
                columns = ['MA{}'.format(i) for i in range(qMin, qMax+1)]
                )
       
        for p,q in itertools.product(range(pMin,pMax+1), range(qMin,qMax+1)):
            if p == 0 and q == 0:
                BICResult.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue
            
            try:
                model = ARIMA(dfTrain[col], order = (p, d, q))
                result = model.fit()
                BICResult.loc['AR{}'.format(p), 'MA{}'.format(q)] = result.bic
            except:
                continue
        
        BICResult = BICResult[BICResult.columns].astype(float)
        
        fig,ax = plt.subplots(figsize = (10,8))
        ax = sns.heatmap(
                BICResult,
                mask = BICResult.isnull(),
                ax = ax,
                annot = True,
                fmt = '.2f',
                )
        ax.set_title('BIC')
        plt.show()
        
# way no.2
        trainResults = arma_order_select_ic(list(dfTrain['S_DQ_ADJCLOSE']), ic = ['aic','bic'], trend = 'nc', max_ar =8, max_ma =8) 
        
        print('AIC', trainResults.aic_min_order)
        print('BIC', trainResults.bic_min_order)
          
        
    def evaluate(dfTest, col, p, q, d):
        model = ARIMA(dfTest[col], order = (p, d, q))
        results = model.fit()
        resid = results.resid
        fig = plt.figure(figsize=(12,8))
        fig = plot_acf(resid.values.squeeze(), lags = 40)
        plt.show()

    def predict(dfTest, col, p, q, d):
         model = ARIMA(dfTest[col], order = (p, d, q))
         results = model.fit()
         predictSunsplots = results.predict(start= '20191101',end= '20191120',dynamic = False)
         print(predictSunsplots)
         fig,ax = plt.subplots(figsize = (12,8))
         dfTest[col].plot(ax = ax)
         predictSunsplots.plot(ax = ax)
         plt.show()
         
         results.forecast()[0]
         
         
if __name__ == '__main__':
    
    defaultCode = '601318'
    parse = OptionParser()
    parse.add_option('-c', '--code', help = 'use code to choose', default = defaultCode)
    options, args = opt.parse()
    ARI = ARI(code)
    df = ARI.loadData()
    dfTrain = df.iloc[0:5*len(df)//6]
    dfTest = df.iloc[5*len(df)//6:len(df)]
    col = 'S_DQ_ADJCLOSE'
    params = ARI.calParams(dfTrain, col)
    p = 5
    #q = 
    d = 2
    p=1
    q =4
    ARI.evaluate(dfTest, p, q, d)
    