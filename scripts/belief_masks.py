#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform
from skimage import measure

import os
import scipy as scp
import scipy.misc

import numpy as npy
import sys
import matplotlib.pyplot as plt
import copy


roadmap = npy.loadtxt('ROADMAP.txt')
threshold = 100

for i in range(0,448):	
	print("Running on Image",i)
	belief_maps = npy.zeros((10,480,640))
	classes = npy.loadtxt("class_{0}.txt".format(i))
	mask = (classes==11)*roadmap
	labels = measure.label(mask)

	label_list = npy.zeros(labels.max()).astype(int)	
	for j in range(labels.max()):
		label_list[j] = ((labels==j)*npy.ones((480,640))).sum()
	sort = npy.argsort(-label_list)
	
	for k in range(1,min(11,sort.shape[0])):		
		if (label_list[sort[k]]>threshold):
			belief_maps[k-1] = (labels==sort[k])*npy.ones((480,640))		

	with file('belief_maps_{0}.txt'.format(i),'w') as outfile:
		for data in belief_maps:
			outfile.write('#Belief_Map.\n')
			npy.savetxt(outfile,data,fmt='%i')


