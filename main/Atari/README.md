# Monitoring the agent's epistemic uncertainty in Atari

This folder contains the code used to produce figure 3 in our paper. Running "train_atari.py" trains the agent and saves its weights in "network.pth", "monitor_episode_uncertainty.py" makes the trained agent play an episode and saves both its uncertainty estimate in "uncertainties" and the game frames in the game_frames folder.

The "make_uncertainty_plot" notebook loads and plots the uncertainties in "uncertainties", producing figure 3.

Part of the code used for these experiments was developed from the OpenAI baselines and the Stable Baselines.
