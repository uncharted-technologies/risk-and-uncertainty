# Atari experiments
Code used to train and test agents on Atari

# Structure

**agent**: the learning agents: DQN, IDS, QR-DQN, and QR-DQN with Thompson sampling (EQRDQN)
**experiments**: containing scripts to train and test agents.
**results**: by default, the output of scripts from the experiments folder is saved here
**plotting**: loads data from the results folder and plots it, yielding the figures in our paper

#Running experiments: example
We provide example scripts for training agents on both Cartpole and Atari.

To train a QR-DQN agent on CartPole, run the following command from this folder:

'''
python -m experiments.qrdqn.train_cartpole
'''

A log folder will automatically be created in the results folder, which at the end of training will contain the network weights of the trained agent, log data and plots of the agent's performance. The trained agent's performance can then be viewed with:

'''
python -m experiments.qrdqn.enjoy_cartpole
'''

For Atari, we have included evaluation scores every 1 million frames in the results folder for IDS, QR-DQN, and QR-DQN with Thompson sampling (EQRDQN) that can be plotted with:

'''
python -m plotting.plot_atari
'''

Note: trained agent weights are quite large and are thus not included in this folder. We can provide agent weights and the script used to produce figure 5 in our paper upon reasonable request.
