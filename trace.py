
# trace catch 
# design an algorithm to catch a trend using as little data as possible
# trend is defined as the similarity of risk(variance) and return(average value)
# use Euclidean distance to show the similarity
# also, in order to escape sharpe change using derivative to help

import numpy as np
import pandas as pd
from scipy import *
from optparse import OptionParser
from sqlalchemy import *
import tensorflow as tf

class trace(error, derivative, length):

    def __init__(self):
        self.error = error
        self.derivative = derivative
        self.length = length

    def loadData():

        return data

    def sharpList():

        return sharpList

    def Edistance():

        return Edistance

    def derivative():

        return derivative

    def handle(self):
        e = self.error
        d = self.derivative
        l = self.length
        data = loadData()

        turningPoints = list()
        for i in range():
            sharpRatios = sharpList(data, l)
            Edistance = Edistance(data, l)
            derivative = derivative(data, l)
            if derivative > d
                turningPoints.append(i)

        if max(Edistance) < e:
        
        

        return l




if __name__ == '__main__':
    opt = OptionParser()
    opt.add_option('-e','--error', help = 'Euclidean distance', dest = 'error', default = 0.1)
    opt.add_option('-d','--derivative', help = 'derivative', dest = 'derivative', default = 1)
    options, args = parse_arg()
    length = 
    trace = trace(options.error, options.derivative, length)
    trace.handle()
