# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:49:31 2019

@author: 86156
"""

class test():
    
    def __init__(self):
        
        return None
    
    def printnum(self):
        a = [2,1,2,3,4]
        print(a[2:4].index(2))
        print('ok')
        

if __name__ == '__main__':
    if 1 in [1,2,3,4]:
        print('true')
    a = [1,2,3,4,5]
    a.remove(3)
    print(a)
    test = test()
    test.printnum()
