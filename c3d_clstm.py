import io
import sys
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.framework import ops
import ConvLSTMCell as clstm


def c3d_clstm(inputs, num_classes, reuse, is_training):
  """Builds the Conv3D-ConvLSTM Networks."""
  #指定机器中使用的GPU
  with tf.device('/gpu:0'):
    with tf.variable_scope('Conv3D_ConvLSTM', reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      if inputs.get_shape().ndims!=5:
        raise Exception("The input dimension of 3DCNN must be rank 5")
      network_input = tl.layers.InputLayer(inputs, name='input_layer')      #Input Layer
      # 3DCNN-BN Layer 1
      #ConvLayer shape=[filter_depth, filter_height, filter_width, in_channels, out_channels]
      #stride= [strides_batch,strides_depth,strides_height,strides_width,strides_channel]
      #stride= [1,depth,stride,stride,1]
      conv3d_1 = tl.layers.Conv3dLayer(network_input,
                                        act=tf.identity,
                                        shape=[3,3,3,3,64],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_1')
      conv3d_1 = tl.layers.BatchNormLayer(prev_layer=conv3d_1,
                                        act=tf.nn.relu,
                                        is_train=is_training,
                                        name='BatchNorm_1')

      #池化层size=[1, width,height, width, 1]
      #stride=[1,depth,stride,stride, 1]
      pool3d_1 = tl.layers.PoolLayer(conv3d_1,
                                        ksize=[1,1,2,2,1],
                                        strides=[1,1,2,2,1],
                                        padding='SAME',
                                        pool = tf.nn.max_pool3d,
                                        name='Pool3D_1')
      # 3DCNN-BN Layer 2
      conv3d_2_3x3 = tl.layers.Conv3dLayer(pool3d_1,
                                        act=tf.identity,
                                        shape=[3,3,3,64,128],
                                        strides=[1,1,1,1,1],
                                        padding='SAME',
                                        name='Conv3d_2_3x3')
      conv3d_2_3x3 = tl.layers.BatchNormLayer(prev_layer=conv3d_2_3x3,
                                              act=tf.nn.relu,
                                              is_train=is_training,
                                              name='BatchNorm_2_3x3')
      pool3d_2 = tl.layers.PoolLayer(conv3d_2_3x3,
                                     ksize=[1,2,2,2,1],
                                     strides=[1,2,2,2,1],
                                     padding='SAME',
                                     pool = tf.nn.max_pool3d,
                                     name='Pool3D_2')
      # 3DCNN-BN Layer 3
      conv3d_3a_3x3 = tl.layers.Conv3dLayer(pool3d_2,
                                            act=tf.identity,
                                            shape=[3,3,3,128,256],
                                            strides=[1,1,1,1,1],
                                            padding='SAME',
                                            name='Conv3d_3a_3x3')
      conv3d_3b_3x3 = tl.layers.Conv3dLayer(conv3d_3a_3x3,
                                            act=tf.identity,
                                            shape=[3,3,3,256,256],
                                            strides=[1,1,1,1,1],
                                            padding='SAME',
                                            name='Conv3d_3b_3x3')
      conv3d_3_3x3 = tl.layers.BatchNormLayer(prev_layer=conv3d_3b_3x3,
                                              act=tf.nn.relu,
                                              is_train=is_training,
                                              name='BatchNorm_3_3x3')
#      pool3d_3 = tl.layers.PoolLayer(conv3d_3_3x3,
#                                        ksize=[1,2,2,2,1],
#                                        strides=[1,2,2,2,1],
#                                        padding='SAME',
#                                        pool = tf.nn.max_pool3d,
#                                        name='Pool3D_3')
      # ConvLstm Layer
      shape3d = conv3d_3_3x3.outputs.get_shape().as_list()
      num_steps = shape3d[1]

      #cell shape width*height,关于cell shell shape的理解还需要查阅相关论文
      #关于shape大小的理解
      convlstm1=tl.layers.ConvLSTMLayer(prev_layer=conv3d_3_3x3,
                                        cell_shape=[28,28],
                                        filter_size=[3,3],
                                        feature_map=256,
                                        initializer=tf.random_uniform_initializer(-0.1,0.1),
                                        n_steps=num_steps,
                                        return_last=False,
                                        return_seq_2d=False,
                                        name='clstm_layer_1')

      convlstm2 = tl.layers.ConvLSTMLayer(prev_layer=convlstm1,
                                          cell_shape=[28,28],
                                          filter_size=[3,3],
                                          feature_map=384,
                                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                          n_steps=num_steps,
                                          return_last=True,
                                          return_seq_2d=False,
                                          name='clstm_layer_2')
      # SPP Layer 1
      spp_bin_1 = tl.layers.PoolLayer(convlstm2,
                                      ksize=[1, 28, 28, 1],
                                      strides=[1, 28, 28, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='SPP_1')
      spp_bin_1 = tl.layers.FlattenLayer(spp_bin_1,
                                         name='Flatten_SPP_1')
      spp_bin_2 = tl.layers.PoolLayer(convlstm2,
                                      ksize=[1, 14, 14, 1],
                                      strides=[1, 14, 14, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='SPP_2')
      spp_bin_2 = tl.layers.FlattenLayer(spp_bin_2,
                                         name='Flatten_SPP_2')
      spp_bin_4 = tl.layers.PoolLayer(convlstm2,
                                      ksize=[1, 7, 7, 1],
                                      strides=[1, 7, 7, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='SPP_4')
      spp_bin_4 = tl.layers.FlattenLayer(spp_bin_4,
                                         name='Flatten_SPP_4')
      spp_bin_7 = tl.layers.PoolLayer(convlstm2,
                                      ksize=[1, 4, 4, 1],
                                      strides=[1, 4, 4, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='SPP_8')
      spp_bin_7 = tl.layers.FlattenLayer(spp_bin_7,
                                         name='Flatten_SPP_7')
      concat_spp = tl.layers.ConcatLayer([spp_bin_1,
                                          spp_bin_2,
                                          spp_bin_4,
                                          spp_bin_7],
                                         concat_dim=1,
                                         name='Concat_SPP')
      # FC Layer 1
      classes = tl.layers.DropconnectDenseLayer(concat_spp,
                                                keep=0.5,
                                                n_units=num_classes,
                                                act=tf.identity,
                                                name='Classes')
    return classes
