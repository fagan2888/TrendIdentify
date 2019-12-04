
# written by zhaoliyuan
# MBSS indicator

import pandas as pd
import numpy as np
from config import *
from sqlalchemy import *
from WindPy import *
from optparse import OptionParser

class MBSS():
    
    def __init__(self, code, sdate, edate):
        
        self.code = code
        self.sdate = sdate
        self.edate = edate
    
    def loadData(self):
        
        code = self.code
        sdate = self.sdate
        edate = self.edate
        
        w.start()
        _,df = w.wsd(code, "dealnum,volume,amt", sdate, edate, usedf = True)
        
        return df
    
    def handle(self,df):
        
        MBSS = list()
        for i in range(len(df)):
            mbss = df['VOLUME'][i]/df['DEALNUM'][i]
            MBSS.append(mbss)
        df['MBSS'] = MBSS
        out = df['MBSS']
           
        return out



if __name__ == '__main__':

#    db = create_engine('wind')
#    meta = MetaData(bind = db)
#    t = Table('ashareeodprices', meta, autoload = True)
#    columns = [
#            t.c.S_INFO_WINDCODE,
#            t.c.TRADE_DT,
#            t.c.S_DQ_CLOSE,
#            t.c.S_DQ_OPEN,
#            t.c.S_DQ_HIGH,
#            t.c.S_DQ_LOW,
#            t.c.S_DQ_PCTCHANGE,
#            ]
#    sql = select()
#    sql = sql.where()
#    df = pd.read_sql(sql,db)
    opt = OptionParser()
    opt.add_option('--code', help = 'stock code', default = '000001.SZ')
    opt.add_option('--sdate', help = 'start date', default = '2019-11-03')
    opt.add_option('--edate', help = 'end date', default = '2019-12-02')
    options, args = opt.parse_args()
    MBSS = MBSS(options.code, options.sdate, options.edate)
    df = MBSS.loadData()
    out = MBSS.handle(df)
