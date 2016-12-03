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

import fcn8_vgg
import utils
import matplotlib.pyplot as plt
import copy

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# img1 = skimage.io.imread("./test_data/tabby_cat.png")
img1 = skimage.io.imread("./test_data/pl.jpg")

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')

    tensors = [vgg_fcn.pred, vgg_fcn.pred_up, vgg_fcn.upscore32]
    down, up, score = sess.run(tensors, feed_dict=feed_dict)

    # print(up[0])
    print(score.shape)
    score_reshape = score.reshape((20,img1.shape[0],img1.shape[1]))

    # Choosing Class k
    # k=4
    img = copy.deepcopy(img1)

    for k in range(0,20):
    # k = 

    # Building the Heatmap
        print(k)
        belief = np.multiply(up[0]==k,score_reshape[k])
        # belief = (up[0]==k)*score_reshape[k]*

        for a in range(0,3):
            img[:,:,a] = (up[0]==k)*img1[:,:,a]
        # img = np.multiply(up[0]==k,img1)   
        belief += belief.min()    
        belief /= belief.sum()


        plt.imshow(img)
        # plt.imshow(belief)
        # plt.colorbar()
        plt.show()

    
    # belief = np.multiply(up[0]==k,score_reshape[k])
    # belief += belief.min()    
    # belief /= belief.sum()

    # plt.imshow(belief)
    # plt.colorbar()
    # plt.show()

    with file('class_predictions.txt','w') as outfile:
        outfile.write('#Class Predictions.\n')
        np.savetxt(outfile,up[0],fmt='%i')

    with file('score_values.txt','w') as outfile:
        for data in score_reshape:
            outfile.write('#Score_Values.\n')
            np.savetxt(outfile,data,fmt='%-7.5f')

    with file('belief.txt','w') as outfile:
        outfile.write('#Belief.\n')
        np.savetxt(outfile,belief,fmt='%-7.5f')

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])        


    scp.misc.imsave('fcn8_downsampled.png', down_color)
    scp.misc.imsave('fcn8_upsampled.png', up_color)    