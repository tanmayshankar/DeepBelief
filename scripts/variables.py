#!/usr/bin/env python
import numpy as npy
import matplotlib.pyplot as plt
import sys
import random
from scipy import signal
import copy

discrete_size_x = 64
discrete_size_y = 48

#Action size also determines number of convolutional filters. 
action_size = 8
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT, NOTHING.

#Transition space size determines size of convolutional filters. 
transition_space = 5
backprop_belief = npy.zeros((discrete_size_y,discrete_size_x))

w = transition_space/2

q_value_estimate = npy.ones((action_size,discrete_size_y,discrete_size_x))
reward_estimate = npy.zeros((action_size,discrete_size_y,discrete_size_x))
q_value_layers = npy.zeros((action_size,discrete_size_y,discrete_size_x))
value_function = npy.zeros((discrete_size_y,discrete_size_x))

qmdp_values = npy.zeros(action_size)
qmdp_values_softmax = npy.zeros(action_size)
target_actions = npy.zeros(action_size)
belief_target_actions = npy.zeros(action_size)

