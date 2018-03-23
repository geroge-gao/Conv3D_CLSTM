from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import pylab as plt

def c3d_clstm(inputs,num_classes,reuse,is_training):

    #指定使用的CPU
    with tf.device("/gpu:0"):
        with tf.variable_scope("C3D_CLstm",reuse=reuse):
            #获取序列模型
            seq=Sequential()
