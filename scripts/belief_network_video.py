#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn16_vgg
import utils
import matplotlib.pyplot as plt
import copy

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

sess = tf.InteractiveSession()
images = tf.placeholder("float")
vgg_fcn = fcn16_vgg.FCN16VGG()

i=0
img1 = skimage.io.imread("image_{0}.png".format(i))      
feed_dict = {images: img1}
batch_images = tf.expand_dims(images, 0)

with tf.name_scope("content_vgg"):
    vgg_fcn.build(batch_images, debug=True)

print('Finished building Network.')

init = tf.initialize_all_variables()
sess.run(tf.initialize_all_variables())

k=11
for i in range(1,448):

# for i in range(0,1):    

    print("Running on Image", i)    
    img1 = skimage.io.imread("image_{0}.png".format(i))  
    
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    tensors = [vgg_fcn.pred, vgg_fcn.pred_up, vgg_fcn.upscore32]
    down, up, score = sess.run(tensors, feed_dict=feed_dict)
        
    score_reshape = score.reshape((20,img1.shape[0],img1.shape[1]))                  

    # Building the Heatmap

    belief = np.multiply(up[0]==k,score_reshape[k])
        # belief = (up[0]==k)*score_reshape[k]*

    for a in range(0,3):
        img[:,:,a] = (up[0]==k)*img1[:,:,a]
        # img = np.multiply(up[0]==k,img1)   
    belief += belief.min()    
    belief /= belief.sum()

    # plt.imshow(img)        
    # plt.show()

    with file('class_{0}.txt'.format(i),'w') as outfile:
        outfile.write('#Class Predictions.\n')
        np.savetxt(outfile,up[0],fmt='%i')

    with file('scores_{0}.txt'.format(i),'w') as outfile:
        for data in score_reshape:
            outfile.write('#Score_Values.\n')
            np.savetxt(outfile,data,fmt='%-7.5f')

    with file('belief_{0}.txt'.format(i),'w') as outfile:
        outfile.write('#Belief.\n')
        np.savetxt(outfile,belief,fmt='%-7.5f')