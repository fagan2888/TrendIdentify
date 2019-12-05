# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:21:48 2019

@author: 86156
"""

import numpy as np
import tensorflow as tf

sess = tf.Session()

# devices that will be used in algorithm
#tf.device('/device:gpu:1')

# data types
a = tf.zeros([2,3])
b = tf.zeros_like(a)
c = tf.to_double(a)
d = tf.cast(c,dtype = 'int64')
e = tf.constant([1,2,3,4], dtype = 'int64', shape = [2,2])
e = tf.convert_to_tensor([1,2],dtype = 'int32',  dtype_hint = 'float64')
a = tf.linspace(start = 1.0, stop = 5.0, num = 10)
t = tf.is_tensor(a)
# use mask to choose some part of the data
a = np.array([[1,2],[2,3],[3,4]])
print(a)
b = tf.boolean_mask(a, mask = [[True,False],[True,False],[True,False]], name = 'b') 
print(b)
# copy 
x = tf.Variable([1,2,3])
y = tf.broadcast_to(x, shape = [4,3])
y = tf.identity(x)
t = tf.identity_n([x,y])
# expand_dims
# change dims of a tensor add 1 dim into it
a = tf.convert_to_tensor([[1,2],[2,3]], dtype = 'float64') # shape a (2,2)
t = tf.expand_dims(a, 0) # shape changed into (1,2,2) the second parameters is the position
# I matrix or a batch of matrices be careful that batch_shape should be a list
t = tf.eye(3)
t = tf.eye(num_rows = 3, num_columns = 4, batch_shape = [2], dtype = 'float64')
# meshgrid
t = tf.meshgrid([1,2,3],[2,3,4])

# math
t = tf.add(1.0,2.0) # plus
t = tf.math.multiply(1.2,3.4) # times
t = tf.matmul(np.mat([1,2]),np.mat([[2],[2]])) # matirx times
t = tf.log(tf.exp(1))# logarithm and exponent
t = tf.subtract(2.0,1.0) # substract
# calculate gradient be careful that x must be set as a tensor or tf.constant
x = [3.0, 2.0]
x = tf.convert_to_tensor(x, dtype = 'float64')
x = tf.constant([1.0,2.0,3.0])
y = x**2+x+1
t = tf.gradients(y, x)

# use tensorflow to define if else function
def f1(): return 1
def f2(): return 2
def f3(): return 3
x = 1
y = 2
fun = tf.case([(tf.less(x,y),f1),(tf.greater(x,y),f2)], default = f3, exclusive = True)

# apply functions on the elments of a list unpacked from dim 0
a = np.array([1,2,3,4,5])
b = (np.array([1,2]),np.array([3,4]))
def fun (x):
    return x*3
t = tf.map_fn(fn = fun, elems = a)
t = tf.map_fn(lambda x: x**2, a)
t = tf.map_fn(lambda x:(x,-x**2), a)
t = tf.map_fn(lambda x: x[0]+x[1], b, dtype = 'int32')


# normalization 
# global_norm = np.sqrt(sum([element**2 for element in x])) 
# x[i] will be set as x[i]*clip_norm/np.max([clip_norm, global_norm])
a = [1.0,2.0,3.0]
b = tf.clip_by_global_norm(a, clip_norm = 2)
# in following case x[i] will be set as x[i]*clip_norm/global_norm
b = tf.clip_by_norm(a, clip_norm = 2)
# in following case x[i] x[i] less than clip_value_min will be set as clip_value_min 
# x[i] greater than clip_value_max will be set as clip_value_max otherwise x[i] will not be changed
b = tf.clip_by_value(a, clip_value_min = 2, clip_value_max = 6)

# cut data into several parts elements in partitions must be int32 and start from 0 
data = [1,2,3,4,5,6]
partitions = [1,1,2,0,0,2]
t = tf.dynamic_partition(data=data,partitions=partitions,num_partitions = 3)
# merge back 
# be careful that elements in indices can not be an integer when nessary used list instead  
data = [[1,3],[2],[4,5,6]]
indices = [[0,2],[1],[3,4,5]]
t = tf.dynamic_stitch(data = data, indices = indices)

# Einstein summation 
# (many kinds of calculations can be realized for example: dot product, outer product, transpose, trace, matirx multiplication)
# mind that input should convert into tensors
a1 = np.ones([3,3])
a2 = np.matrix([[1,2],[2,3],[4,5]])
a3 = np.ones([2,2])
a1 = tf.convert_to_tensor(a1, dtype = 'float64')
a2 = tf.convert_to_tensor(a2, dtype = 'float64')
a3 = tf.convert_to_tensor(a3, dtype = 'float64')
v1 = tf.convert_to_tensor([1,2,3], dtype = 'float64')
v2 = tf.convert_to_tensor([3,6,9], dtype = 'float64')
# matix multiplication
t = tf.einsum('ij,jk->ik',a1,a2)
# dot product
t = tf.einsum('i,i->',v1,v2)
# outer product
t = tf.einsum('i,j->ij',v1,v2)
# trace
t = tf.einsum('ii',a1)

# gather slices
a = [1,2,3]
b = [2,3,4]
t = tf.gather(params = [a,b], indices = [0,1])
# gather slices from params into a tensor and choose some part of it to form a new tensor
a = [1,2,3]
b = [2,3,4]
t = tf.gather_nd(params = [a,b], indices = [[0],[1]])

# work out the hessian matirces
x = tf.convert_to_tensor([[1],[2],[3],[4],[5]], dtype = 'float64')
x = tf.convert_to_tensor([1,2,3,4,5], dtype = 'float64')
y = tf.log(x)
t = tf.hessians(ys = y, xs =x)

# workout a histogram
# nbins of groups will be created: step = (value_range[1]-value_range[0])/nbins
# [-inf, value_range[0]+step),......,[value_range[1]-step,inf)
x = tf.random_normal(shape = [1,100], mean = 0, stddev = 2, dtype = 'float32')
t = tf.histogram_fixed_width(values = x, value_range = [-6,6], nbins = 10, dtype = 'int64')
# bin the given values into a histogram
# return a vector with same length of x 
# eatch elements in vector means the group that a value belongs to 0 is the start
t = tf.histogram_fixed_width_bins(values = x, value_range = [-6.0,6.0], nbins = 10, dtype = 'float32')

# add index in a tensor(unknown)
x = tf.convert_to_tensor([[1,2],[2,3],[3,4]])
t = tf.IndexedSlices(values = x, indices = tf.convert_to_tensor(['a','b','c']))

# create a callable graph from a python function


###############################################################################
###### the following part is about ANN based on tensorflow ######

## use keras to build ANN: tf.keras. ##
# start a linear stack of layers use: tf.keras.Sequential()

# built-in activation functions use: tf.keras.activations.

# canned architecture with pre-trained weights use: tf.keras.applications.
# densenet modules use: tf.keras.applications.densenet.
# imagenet utilities use: tf.keras.applications.imagenet_utils.
# inception-resnet user: tf.keras.applications.inception_resnet_v2.
# etc use tf.keras.applications. to find more pre-trained network

# define a network more specifically use: tf.keras.backend. to change details in a network

# convolution network : tf.keras.backend.conv1d conv2d conv3d

# impose constraints on weight values: tf.keras.constraints.

# call certain points during model traininig use: tf.keras.callbacks.


## wrappers for primitive Neural Net Operations : tf.nn. ##
# classes for RNN: tf.nn.RNNCellDeviceWrapper. tf.nn.RNNCellDropoutWrapper. tf.nn.RNNCellResidualWrapper

# convultion RNN: tf.nn.conv1d 

# activations: tf.nn.sigmoid tanh relu selu softmax ......

# optimizers: tf.keras.optimizers.Adam RMSprop SGD ......


## others ##
# tf.test
# tf.train


###############################################################################
# example
from tensorflow.examples.tutorials import mnist
import time 

mnistData = mnist.input_data.read_data_sets('E:/codes/tfboys/mnistfata/' , one_hot = True)
trainData = mnistData.train
testData = mnistData.test

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides= [1,1,1,1], padding = 'SAME')

def maxPooling(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

start = time.clock()

x = tf.placeholder(tf.float32, (None,784))
xImage = tf.reshape(x, [-1,28,28,1])

wConv1 = weightVariable([5,5,1,32])
bConv1 = biasVariable([32])
hConv1 = tf.nn.relu(conv2d(xImage, wConv1)+bConv1)
hPool1 = maxPooling(hConv1)

wConv2 = weightVariable([5,5,32,64])
bConv2 = biasVariable([64])
hConv2 = tf.nn.relu(conv2d(hPool1, wConv2)+bConv2)
hPool2 = maxPooling(hConv2)

wFc1 = weightVariable([7*7*64, 1024])
bFc1 = biasVariable([1024])
hPool2Flat = tf.reshape(hPool2, [-1, 7*7*64])
hFc1 = tf.nn.relu(tf.matmul(hPool2Flat, wFc1)+bFc1)

keepProb = tf.placeholder('float')
hFc1Drop = tf.nn.dropout(hFc1, keepProb)

wFc2 = weightVariable([1024,10])
bFc2 = biasVariable([10])
yConv = tf.nn.softmax(tf.matmul(hFc1Drop, wFc2)+bFc2)

y_ = tf.placeholder('float', [None, 10])
crossEntropy = -tf.reduce_sum(y_ * tf.log(yConv))
trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(yConv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, 'float'))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(500):
    batch = trainData.next_batch(50)
    if i % 100 == 0:
        trainAccuracy = accuracy.eval(
                session = sess, 
                feed_dict = {x:batch[0], y_:batch[1], keepProb: 1.0}
                )
        print('step %d, trainAccuracy %g'%(i, trainAccuracy))
    trainStep.run(
            session = sess, 
            feed_dict = {x:batch[0], y_:batch[1], keepProb: 0.5}
            )
print('test accuracy %g'%accuracy.eval(
        session = sess, 
        feed_dict = {x:testData.images, y_:testData.labels, keepProb: 1.0}
        ))
end = time.clock()
print('running time is %g s'%(end-start))

sess.run(t.values)