# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:49:31 2019

@author: 86156
"""

import pandas as pd
import numpy as np
from ipdb import set_trace

class test():
    
    def __init__(self):
        
        return None
    
    def printnum(self):
        a = [2,1,2,3,4]
        print(a[2:4].index(2))
        print('ok')
        

if __name__ == '__main__':
#    if 1 in [1,2,3,4]:
#        print('true')
#    a = [1,2,3,4,5]
#    print(a)
#    a.remove(3)
#    print(a)
#    del(a[a.index(2)])
#    print(a)
#    test = test()
#    test.printnum()

    df = pd.DataFrame({'a':['1','1','2','2','3','3','4','4'],'b':[1,2,1,2,1,2,1,3],'c':[1,3,1,4,1,5,1,6]})
    print(df.shape[1])
    set_trace()
    df = df.set_index(['a','b'])
#    dfNew = pd.DataFrame({'a1':['1','1','2','2','3','3','4','4'],'b1':[1,2,1,2,1,2,1,3],'c':[1,3,1,4,1,5,1,6]})
#    dfNew
    print(df.index.get_level_values(0).values)
#    print(df.set_index('a').stack())
#    df = df.groupby('a').get_group('1')
#    print(df.shape)

#    a = np.ones((1,2,3))
#    b = a[0]
#    print(b)

    

