# Exploration using Thompson Sampling in Atari

This folder contains the code used to produce figure 4 in the supplementary information of our paper. Experimental results are loaded in the "plot_results" notebook, which produces the graphs shown in figure 4.

The two subfolders contain the code used to implement QR-DQN with the esilon-greedy and Thompson sampling policies. In both subfolders, agents are trained by running the "train_atari" scripts. Agent weights are saved every 1M frames, and evaluation scores are produced by running "evaluate_agent".

Part of the code used for these experiments was developed from the OpenAI baselines (https://github.com/openai/baselines) and the Stable Baselines (https://github.com/hill-a/stable-baselines).
