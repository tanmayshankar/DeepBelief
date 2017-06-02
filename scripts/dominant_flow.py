#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform
from skimage import measure

import os
import scipy as scp
import scipy.misc

import numpy as np
import sys
import matplotlib.pyplot as plt
import copy


roadmap = npy.loadtxt('ROADMAP.txt')
angle_range = npy.linspace(0,360,9)-180

# action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1])

for i in range(0,448):	
	print("Running on Image",i)

	belief_maps = npy.loadtxt("../CARS_10/belief_maps_{0}.txt".format(i)).reshape(10,480,640)	
	flow = npy.loadtxt("flow_{0}.txt".format(i)).reshape(480,640,2)
	action_map = npy.zeros(10)

	for k in range(0,10):
		fy = (belief_maps[k]*flow[:,:,0]).sum()
		fx = (belief_maps[k]*flow[:,:,1]).sum()

		ang = npy.arctan2(fy,fx)*(180/npy.pi)	

		if (ang>-22.5)and(ang<22.5):
			action_map[k] = 1
		if (ang>22.5)and(ang<67.5):
			action_map[k] = 7
		if (ang>67.5)and(ang<112.5):
			action_map[k] = 3
		if (ang>112.5)and(ang<157.5):
			action_map[k] = 5		
		if (ang<-22.5)and(ang>-67.5):
			action_map[k] = 6
		if (ang<-67.5)and(ang>-112.5):
			action_map[k] = 2
		if (ang<-112.5)and(ang>-157.5):
			action_map[k] = 4							
		if (ang>157.5)or(ang<-157.5):
			action_map[k] = 0

	with file('action_map_{0}.txt'.format(i),'w') as outfile:
		outfile.write('#Action_Map.\n')
		npy.savetxt(outfile,action_map,fmt='%i')


