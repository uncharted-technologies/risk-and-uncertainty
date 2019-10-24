## Code to reproduce the contextual bandit experiments

This folder is organized as follows :

* A data folder where you can find the **mushroom data** and the **outputs of the "run" files** (run_bayes_by_backprop.py, run_eqrdqn.py, run_greedy.py, and run_two_networks.py) 
* A functions folder which contains the models to make the different plots.
* 4 scripts to run the models and output the different plots (run_bayes_by_backprop.py, run_eqrdqn.py, run_greedy.py, and run_two_networks.py). 
* A notebook cumulative_regrets.ipynb to plot the results.

To reproduce the results, you first have to :

1) Run the run files,
2) Create a dictionary {name : array of regret results (n_seeds, num_frames)}
3) Run the notebook to plot the results
