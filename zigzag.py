
# zigzag algorithm
# created by zhaoliyuan

import numpy as np
import pandas as pd
from optparse import OptionParser
from ipdb import set_trace

class zigzag():

    def __init__(self, depth, backstep, deviation, choice):
        self.depth = depth
        self.backstep = backstep
        self.deviation = deviation
        self.choice = choice

    def handle(self, data):
        depth = self.depth
        backstep = self.backstep
        deviation = self.deviation
        choice = self.choice
        
        lows = list()
        highs = list()
        indicesLow = list()
        indicesHigh = list()
        # initialize
        low = min(data[0:int(depth)])
        lows.append(low)
        indicesLow.append(data.index(low))
        high = max(data[0:int(depth)])
        highs.append(high)
        indicesHigh.append(data.index(high))

        flag = 1 - choice
        for i in range(int(depth),int(len(data)-depth)):
            if flag == 0:
                # find next low point
                if lows[-1] > data[i]:
                    low = data[i]
                    lows.append(low)
                    indicesLow.append(i)
                    flag = 1
                    index = data.index(lows[0])
                    # delete the low point that is not in comparison zone
                    if index <= i-depth:
                        del(lows[0]) 
                    # if the point is extremely low
                    if min(lows) - low > deviation:
                        for j in range(i-backstep,i):
                            if data[j] > low and lows[-1] == data[j]:
                                del(lows[-1])
            else:
                # find next high point
                if highs[-1] < data[i]:
                    high = data[i]
                    highs.append(high)
                    indicesHigh.append(i)
                    flag = 0
                    index = data.index(highs[0])
                    # delete the point that is not in comparison zone
                    if index <= i-depth:
                        del(highs[0])
                    # if the point is extremely high
                    if high - max(highs) > deviation:
                        for j in range(i-backstep,i):
                            if data[j] < high and highs[-1] == data[j]:
                                del(highs[-1])


if __name__ == '__main__':
  
    opt = OptionParser()
    opt.add_option('--depth', help = 'the interval for selecting extreme point', type = 'int',default = 12)
    opt.add_option('--backstep',help = 'number of extreme point that will delete from list before a new extreme point, when the new point is max or min in a certain interval', type = 'int', default = 3)
    opt.add_option('--deviation', help = 'difference used to define a new extreme point as a max or min point', type = 'int', default = 5)
    opt.add_option('--choice', help = 'see the start point as a high point or not, 1 stands for high point and 0 stands for low point', type = 'int', default = 0)
    options,args = opt.parse_args()
    zigzag = zigzag(options.depth, options.backstep, options.deviation, options.choice)
    data = list(pd.read_excel('windData.xlsx',index_col = 0)['CLOSE'])
    print(data)
    set_trace()
    zigzag.handle(data)
