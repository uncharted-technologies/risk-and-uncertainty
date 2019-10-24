# Bandit experiments
Code used to train and test agents on the Bandit problem, with results shown in figure 2 of our paper.

# Structure

**agent**: the learning agents: Bayes by Backprop, QR-DQN with an epsilon-greedy policy, and QR-DQN with Thompson sampling

**experiments**: containing scripts to train and test agents.

**results**: by default, the output of scripts from the experiments folder is saved here

**plotting**: loads data from the results folder and plots it, yielding the figures in our paper

# Running experiments: example
We provide example scripts for training agents.

To train a QR-DQN agent with Thompson sampling, run the following command from this folder:

```
python -m experiments.train_thompson
```

A log folder will automatically be created in the results folder, which at the end of training will contain the network weights of the trained agent, log data and plots of the agent's performance. Once several agents can be trained, their cumulative regret curves can be visualized with the following command:

```
python -m plotting.plot_regrets
```
