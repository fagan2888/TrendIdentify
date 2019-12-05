
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
        
        peaks = list()
        indices = list()
        # initialize
        if choice == 1:
            high = max(data[0:int(depth)])
            peaks.append(high)
            indices.append(data[0:int(depth)].index(high))
        else:
            low = min(data[0:int(depth)])
            peaks.append(low)
            indices.append(data[0:int(depth)].index(low))

        flag = choice
        i = indices[-1]
        while i < int(len(data)-depth):
            i = indices[-1]

            # when next value is bigger(smaller) but we want to find a small(big) number
            # use this value to set the last number
            if flag == 0 and data[i+1] > data[i]:
                indices[-1] = i + 1
                peaks[-1] = data[i+1]
                continue

            if flag == 1 and data[i+1] < data[i]:
                indices[-1] = i + 1
                peaks[-1] = data[i+1]
                continue

            if flag == 0:
                # find next low point
                if len(peaks) <= 3:
                    if peaks[-1] > min(data[i:i+depth-1]):
                        low = min(data[i:i+depth-1])
                        peaks.append(low)
                        indices.append(data[i:i+depth-1].index(low)+i)
                        i = indices[-1]
                        flag = 1 - flag
                    else:
                        i = 1 + i
                    print(peaks)
                    print(indices)
                    print('l0')
                    set_trace()
                else:
                    indices = indices[0:len(indices)-2]
                    peaks = peaks[0:len(peaks)-2]
                    i = indices[-1]
                    print(i)
                    low = min(data[i:i+depth-1])
                    peaks.append(low)
                    indices.append(data[i:i+depth-1].index(low)+i)
                    flag = 1 - flag
                    print(peaks)
                    print(indices)
                    print('l')
                    set_trace()
                    # if the point is extremely low
                    if len(peaks) > backstep and peaks[-3] - peaks[-1] > deviation:
                        for j in range(indices[-1]-backstep,indices[-1]):
                            if (data[j] > peaks[-1]) and (j in indices) and ((len(indices)-indices.index(j))%2 == 1):
                                del(peaks[indices.index(j)])
                                indices.remove(j)
                        print(peaks)
                        print(indices)
                        print('el')
                        set_tracce()
            else:
                # find next high point
                if len(peaks) <= 3:
                    if peaks[-1] < max(data[i:i+depth-1]):
                        high = max(data[i:i+depth-1])
                        peaks.append(high)
                        indices.append(data[i:i+depth-1].index(high)+i)
                        flag = 1 -flag
                        i = indices[-1]
                    else:
                        i = 1 + i
                    print(peaks)
                    print(indices)
                    print('h0')
                    set_trace()
                else:
                    indices = indices[0:len(indices)-2]
                    peaks = peaks[0:len(peaks)-2]
                    i = indices[-1]
                    print(i)
                    high = max(data[i:i+depth-1])
                    peaks.append(high)
                    indices.append(data[i:i+depth-1].index(high)+i)
                    flag = 1 - flag
                    print(peaks)
                    print(indices)
                    print('h')
                    set_trace()
                    # if the point is extremely high
                    if len(peaks) > backstep and peaks[-1] - peaks[-3] > deviation:
                        for j in range(indices[-1]-backstep,indices[-1]):
                            if (data[j] < peaks[-1]) and (j in indices) and ((len(indices)-indices.index(j))%2 == 1):
                                del(peaks[indices.index(j)])
                                indices.remove(j)
                        print(peaks)
                        print(indices)
                        print('eh')
                        set_trace()

        return peaks, indices


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
    peaks, indices = zigzag.handle(data)
    print(peaks)
    print(indices)
