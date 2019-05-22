## Code to reproduce the contextual bandit experiments

This folder is organized as follows :

* A data folder where you can find the **mushroom data** and the **outputs of the "run" files** (run_bayes_by_backprop.py and run_eqrdqn_bandits.py) 
* A functions folder which contains the models to make the different plots.
* 2 scripts run_bayes_by_backprop.py and run_eqrdqn_bandits.py to run the models and output the different plots. 
* A notebook cumulative_regrets.ipynb to plot the results.

To reproduce the results, you first have to :

1) Run the run files,
2) Create a dictionnary {name : array of regret results (n_seeds, num_frames)}
3) Run the notebook to plot the results