# TCP-RL
TCP congestion control using reinforcement learning

## Tools
The following RL and networking toolsets are used for the implementation of the project.

### ns-3
ns-3 is a discrete-event network simulator for networking systems, primarily used for research and education. It became a standard in networking research in recent years as the results obtained are accepted by the science community. In this project, we used NS-3 to generate traffic and also simulate our environment.
See: https://www.nsnam.org

### Gym
Gym is an open source Python library, developed by OpenAI, for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Since its release, Gym's API has become the field standard for doing this.
See: https://gym.openai.com/

### ns3-gym
ns3-gym is a framework that integrates both OpenAI's Gym and ns-3 in order to encourage usage of reinforcement learning in networking research. ns3-gym works as a proxy between the ns-3 environment simulator, coded in C++, and the RL agent, coded in Python.
See: https://github.com/tkn-tub/ns3-gym

## Agent
The agent interacts with the network environment and keeps exploring the optimal policy by taking various actions such as increasing or decreasing congestion window size. The environment setup consists of a combination of the ns-3 network simulator, ns3-gym proxy, and a Python based RL agent. Our mapping approach is as follows:

* State: The state space is the profile of the networking simulation that is provided by the ns-3 simulator. It includes attributes such as congestion window size, segment size, segments acknowledged, average round trip time (RTT), and other such connection parameters.
* Action: The actions are configured to increase, decrease or maintain the congestion window size and slow start threshold by certain degrees based on fine-tuning.
* Reward: The reward is a utility function that considers the average RTT of each step as the defining parameter. It allocates the reward depending on the trend in the parameter.

## Installation and testing

* Install Python
* Install and build ns3-gym (this includes ns-3 and dependencies)
* Install Tensorflow and other agent dependencies
* Build the project

Then, in two terminals, run:
```
# Terminal 1:
./waf --run "TCP-RL --transport_prot=TcpRlTimeBased"

# Terminal 2:
./TCP-RL-Agent.py --start=0
```
