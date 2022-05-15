#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

from ns3gym import ns3env
from tcp_base import TcpTimeBased, TcpEventBased

try:
	w_file = open('run.log', 'w')
except:
	w_file = sys.stdout
parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
					type=int,
					default=1,
					help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
					type=int,
					default=1,
					help='Number of iterations, Default: 1')
parser.add_argument('--steps',
					type=int,
					default=100,
					help='Number of steps, Default 100')
args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)

port = 5555
simTime = maxSteps / 10.0 # seconds
seed = 12
simArgs = {"--duration": simTime,}

dashes = "-"*18
input("[{}Press Enter to start{}]".format(dashes, dashes))

# create environment
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed, simArgs=simArgs)

ob_space = env.observation_space
ac_space = env.action_space

# TODO: right now, the next action is selected inside the loop, rather than using get_action.
# this is because we use the decaying epsilon-greedy algo which needs to use the live model
# somehow change or put that logic in an `RLTCP` class that inherits from the Tcp class, like in tcp_base.py,
# then move the class to tcp_base.py and use that agent here
def get_agent(state):
	socketUuid = state[0]
	tcpEnvType = state[1]
	tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
	if tcpAgent is None:
		# get a new agent based on the selected env type
		if tcpEnvType == 0:
			# event-based = 0
			tcpAgent = TcpEventBased()
		else:
			# time-based = 1
			tcpAgent = TcpTimeBased()
		tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
		get_agent.tcpAgents[socketUuid] = tcpAgent

	return tcpAgent

# initialize agent variables
# (useless until the above todo is fixed)
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

def modeler(input_size, output_size):
	"""
	Designs a fully connected neural network.
	"""
	model = tf.keras.Sequential()

	# input layer
	model.add(tf.keras.layers.Dense((input_size + output_size) // 2, input_shape=(input_size,), activation='relu'))

	# output layer
	# maps previous layer of input_size units to output_size units
	# this is a classifier network
	model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
	
	return model

state_size = ob_space.shape[0] - 4 # ignoring 4 env attributes

action_size = 3
action_mapping = {} # dict faster than list
action_mapping[0] = 0
action_mapping[1] = 600
action_mapping[2] = -150

# build model
model = modeler(state_size, action_size)
model.compile(
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
model.summary()

# initialize decaying epsilon-greedy algorithm
# fine-tune to ensure balance of exploration and exploitation
epsilon = 1.0
epsilon_decay_param = iterationNum * 2
min_epsilon = 0.1
epsilon_decay = (((epsilon_decay_param*maxSteps) - 1.0) / (epsilon_decay_param*maxSteps))

# initialize Q-learning's discount factor
discount_factor = 0.95

rewardsum = 0
rew_history = []
cWnd_history = []
pred_cWnd_history = []
rtt_history = []
tp_history = []

recency = maxSteps // 15


done = False

pretty_slash = ['\\', '|', '/', '-']

for iteration in range(iterationNum):
	# set initial state
	state = env.reset()
	# ignore env attributes: socketID, env type, sim time, nodeID
	state = state[4:]

	cWnd = state[1]
	init_cWnd = cWnd

	state = np.reshape(state, [1, state_size])
	try:
		for step in range(maxSteps):
			pretty_index = step % 4
			print("\r{}\r[{}] Logging to file {} {}".format(
				' '*(25+len(w_file.name)),
				pretty_slash[pretty_index],
				w_file.name,
				'.'*(pretty_index+1)
			), end='')

			print("[+] Step: {}".format(step+1), file=w_file)

			# Epsilon-greedy selection
			if step == 0 or np.random.rand(1) < epsilon:
				# explore new situation
				action_index = np.random.randint(0, action_size)
				print("\t[*] Random exploration. Selected action: {}".format(action_index), file=w_file)
			else:
				# exploit gained knowledge
				action_index = np.argmax(model.predict(state)[0])
				print("\t[*] Exploiting gained knowledge. Selected action: {}".format(action_index), file=w_file)

			# Calculate action
			calc_cWnd = cWnd + action_mapping[action_index]

			# Config 1: no cap
			# new_cWnd = calc_cWnd

			# Config 2: cap cWnd by half upon congestion
			# ssThresh is set to half of cWnd when congestion occurs
			# prevent new_cWnd from falling too low
			# ssThresh = state[0][0]
			# new_cWnd = max(init_cWnd, (min(ssThresh, calc_cWnd)))

			# Config 3: if throughput cap detected, fix cWnd
			# detect cap by checking if recent variance less than 1% of current 
			thresh = state[0][0] # ssThresh
			if step+1 > recency:
				tp_dev = math.sqrt(np.var(tp_history[(-recency):]))
				tp_1per = 0.01 * throughput
				if tp_dev < tp_1per:
					thresh = cWnd
			new_cWnd = max(init_cWnd, (min(thresh, calc_cWnd)))
			
			# Config 4: detect throughput cap by checking against experimentally determined value
			# thresh = state[0][0] # ssThresh
			# if step+1 > recency:
			# 	if throughput > 216000: # must be tuned based on bandwidth
			# 		thresh = cWnd
			# new_cWnd = max(init_cWnd, (min(thresh, calc_cWnd)))

			new_ssThresh = int(cWnd/2)
			actions = [new_ssThresh, new_cWnd]

			# Take action step on environment and get feedback
			next_state, reward, done, _ = env.step(actions)

			rewardsum += reward

			next_state = next_state[4:]
			cWnd = next_state[1]
			rtt = next_state[7]
			throughput = next_state[11]
			
			print("\t[#] Next state: ", next_state, file=w_file)
			print("\t[!] Reward: ", reward, file=w_file)

			next_state = np.reshape(next_state, [1, state_size])
			

			# Train incrementally
			# DQN - function approximation using neural networks
			target = reward
			if not done:
				target = (reward + discount_factor * np.amax(model.predict(next_state)[0]))
			target_f = model.predict(state)
			target_f[0][action_index] = target
			model.fit(state, target_f, epochs=1, verbose=0)

			# Update state
			state = next_state

			if done:
				print("[X] Stopping: step: {}, reward sum: {}, epsilon: {:.2}"
						.format(step+1, rewardsum, epsilon),
						file=w_file)
				break

			if epsilon > min_epsilon:
				epsilon *= epsilon_decay

			# Record information
			rew_history.append(rewardsum)
			rtt_history.append(rtt)
			cWnd_history.append(cWnd)
			tp_history.append(throughput)

		print("\n[O] Iteration over.", file=w_file)
		print("[-] Final epsilon value: ", epsilon, file=w_file)
		print("[-] Final reward sum: ", rewardsum, file=w_file)
		print()

	finally:
		print()
		if iteration+1 == iterationNum:
			break
		# if str(input("[?] Continue to next iteration? [Y/n]: ") or "Y").lower() != "y":
		# 	break

mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(2, 2, figsize=(4,2))
plt.tight_layout(pad=0.3)

ax[0, 0].plot(range(len(cWnd_history)), cWnd_history, marker="", linestyle="-")
ax[0, 0].set_title('Congestion windows')
ax[0, 0].set_xlabel('Steps')
ax[0, 0].set_ylabel('CWND (segments)')

ax[0, 1].plot(range(len(tp_history)), tp_history, marker="", linestyle="-")
ax[0, 1].set_title('Throughput over time')
ax[0, 1].set_xlabel('Steps')
ax[0, 1].set_ylabel('Throughput (bits)')

ax[1, 0].plot(range(len(rtt_history)), rtt_history, marker="", linestyle="-")
ax[1, 0].set_title('RTT over time')
ax[1, 0].set_xlabel('Steps')
ax[1, 0].set_ylabel('RTT (microseconds)')

ax[1, 1].plot(range(len(rew_history)), rew_history, marker="", linestyle="-")
ax[1, 1].set_title('Reward sum plot')
ax[1, 1].set_xlabel('Steps')
ax[1, 1].set_ylabel('Accumulated reward')

plt.savefig('plots.png')
plt.show()

