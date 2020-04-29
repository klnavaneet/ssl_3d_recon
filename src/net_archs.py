import sys, os
import tensorflow as tf
from os.path import abspath

import tflearn
from tflearn.layers.conv import conv_2d
from tflearn.layers.conv import conv_2d_transpose
from tflearn.layers.core import fully_connected
import numpy as np


def recon_net_tiny_rgb_skipconn(img_inp, args, return_feat=False):
    feat_dict = {}
    x=img_inp

    # Structure branch
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['conv1'] = x
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['conv2'] = x
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['conv3'] = x
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['conv4'] = x
    #8 8
    x=tflearn.layers.core.fully_connected(x,args.bottleneck,activation='relu',weight_decay=1e-3,regularizer='L2')
    enc_pcl = x
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x1=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x1,args.N_PTS*3,activation='linear',weight_decay=1e-3,regularizer='L2')
    x=tf.reshape(x,(-1,args.N_PTS,3))

    # Feature branch
    #128 128
    y=tflearn.layers.conv.conv_2d(img_inp,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['color_conv1'] = y
    #64 64
    y=tflearn.layers.conv.conv_2d(y,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    feat_dict['color_conv2'] = y
    y=tflearn.layers.core.fully_connected(y,args.bottleneck,activation='relu',weight_decay=1e-3,regularizer='L2')
    enc_feat = tf.concat([enc_pcl, y], axis=-1)
    y=tflearn.layers.core.fully_connected(y,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    y=tflearn.layers.core.fully_connected(y,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    y = tf.concat([x1,y], axis=-1)
    y=tflearn.layers.core.fully_connected(y,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    y=tflearn.layers.core.fully_connected(y,args.N_PTS*3,activation='linear',weight_decay=1e-3,regularizer='L2')
    y=tf.reshape(y,(-1,args.N_PTS,3))
    y = tf.nn.sigmoid(y)

    # Structure + Feature
    z = tf.concat([x,y], axis=-1)
    if not return_feat:
        return x, y
    else:
#        return feat_dict
        return x, y, enc_feat


def pose_net(img_inp, args):
    x=img_inp
    #128 128
    x1=x
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x2=x
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x3=x
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x4=x
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

    # Decoder
    x=tflearn.layers.core.fully_connected(x,args.bottleneck,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,2,activation='linear',weight_decay=1e-3,regularizer='L2')
    # Normalize to range [-pi,pi]
    x = np.pi*tf.nn.tanh(x)
    x = tf.reshape(x,(-1,2))
    return x
