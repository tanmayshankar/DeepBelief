#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import sys
import random
from scipy import signal
import copy
import cv2


discrete_size_x = 64
discrete_size_y = 48

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT, NOTHING.

#Transition space size determines size of convolutional filters. 
transition_space = 5
backprop_belief = npy.zeros((discrete_size_y,discrete_size_x))

trans_mat = npy.loadtxt("trans_model.txt").reshape(8,5,5)

w = transition_space/2

q_value_estimate = npy.ones((action_size,discrete_size_y,discrete_size_x))
reward_estimate = npy.zeros((action_size,discrete_size_y,discrete_size_x))
q_value_layers = npy.zeros((action_size,discrete_size_y,discrete_size_x))
value_function = npy.zeros((discrete_size_y,discrete_size_x))

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)
target_actions = npy.zeros(action_size)
belief_target_actions = npy.zeros(action_size)
learning_rate = 0.5

def calc_softmax():
	global qmdp_values, qmdp_values_softmax

	for act in range(0,action_size):
		qmdp_values_softmax[act] = npy.exp(qmdp_values[act]) / npy.sum(npy.exp(qmdp_values), axis=0)
		

def belief_update_QMDP_values():
	global to_state_belief, q_value_estimate, qmdp_values, from_state_belief, backprop_belief

	for act in range(0,action_size):
		qmdp_values[act] = npy.sum(q_value_estimate[act]*backprop_belief)

def belief_reward_backprop():
	global reward_estimate, qmdp_values_softmax, target_actions, from_state_belief
	global time_index, backprop_belief

	# update_QMDP_values()
	belief_update_QMDP_values()
	calc_softmax()

	# alpha = learning_rate - annealing_rate*time_index
	alpha = learning_rate

	for act in range(0,action_size):
		reward_estimate[act,:,:] -= alpha * (qmdp_values_softmax[act]-belief_target_actions[act]) * backprop_belief[:,:]
		# reward_estimate[act,:,:] -= alpha * (qmdp_values_softmax[act]-target_actions[act]) * backprop_belief[:,:]

def update_q_estimate():
	global reward_estimate, q_value_estimate
	q_value_estimate = reward_estimate + q_value_layers

def beta_update_q_estimate():
	global reward_estimate, q_value_estimate, q_value_layers
	beta = 0.99
	q_value_estimate = (1-beta)*reward_estimate + beta*q_value_layers

def max_pool():
	global q_value_estimate, value_function
	value_function = npy.amax(q_value_estimate, axis=0)

def conv_layer():	
	global value_function, q_value_layers
	trans_mat_flip = copy.deepcopy(trans_mat)

	for act in range(0,action_size):		
		#Convolve with each transition matrix.
		trans_mat_flip[act] = npy.flipud(npy.fliplr(trans_mat[act]))
		q_value_layers[act]=signal.convolve2d(value_function,trans_mat_flip[act],'same','fill',0)

def backprop():	
	belief_reward_backprop()	
	# beta_update_q_estimate()
	update_q_estimate()

def feedback():
	max_pool()
	conv_layer()

# for i in range(0,447):
image_list = range(447)
counter = 0
while image_list:

	i = random.choice(image_list)
	image_list.remove(i)
	counter +=1
	print("Running on Image",i)
	print("Counter",counter)


	score = npy.loadtxt('scores_{0}.txt'.format(i)).reshape(20,480,640)
	bmap = npy.loadtxt('belief_maps_{0}.txt'.format(i)).reshape(10,480,640)
	action_map = npy.loadtxt('action_map_{0}.txt'.format(i))

	for k in range(0,10):

		# Parsing Data.
		# belief = abs(score[11]*bmap[k])
		belief = bmap[k]
		belief = cv2.resize(belief,dsize=(64,48))
		if (belief.sum()>0):
			belief /= belief.sum()

		target_actions[:]=0
		target_actions[action_map[k]]=1

		backprop_belief = copy.deepcopy(belief)

		backprop()
		feedback()


with file('Q_Value_Estimate.txt','w') as outfile:
	for data_slice in q_value_estimate:
		outfile.write('#Q_Value_Estimate.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.5f')

with file('Reward_Function_Estimate.txt','w') as outfile:
	for data_slice in reward_estimate:
		outfile.write('#Reward_Function_Estimate.\n')
		npy.savetxt(outfile,data_slice,fmt='%-7.5f')

with file('Value_Function_Estimate.txt','w') as outfile:
	outfile.write('#Value_Function_Estimate.\n')
	npy.savetxt(outfile,value_function,fmt='%-7.5f')



