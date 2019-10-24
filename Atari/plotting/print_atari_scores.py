"""
Prints best agent performance.
"""

import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pickle
import numpy as np
import matplotlib.pyplot as plt

games = ['alien','amidar','assault','asterix']

ide_seeds = [1,2,3]
qrdqn_seeds = [1,2]
eqrdqn_seeds = [1]
learners = ['IDS','QRDQN']

for idx,game in enumerate(games):

    game_scores_min = []
    game_scores_max = []
    game_scores_mean = []

    game_scores = []

    for learner in learners:

        if learner == 'IDS':
            seeds = ide_seeds
        elif learner == 'QRDQN':
            seeds = qrdqn_seeds
        else:
            seeds = eqrdqn_seeds

        avg_scores = []

        learner_scores = []

        for seed in seeds:

            filename = 'results/Atari/summary/{}/{}_scores{}'.format(learner,game,str(seed))
            scores = np.array(pickle.load(open(filename, 'rb')))
            scores = savgol_filter(scores,5,3)
            learner_scores.append(scores)
        
        game_scores_mean.append(np.stack(learner_scores).mean(axis=0))
        game_scores_min.append(np.stack(learner_scores).min(axis=0))
        game_scores_max.append(np.stack(learner_scores).max(axis=0))

        game_scores.append(np.stack(learner_scores).max(axis=1).mean())

    game_scores_mean = np.stack(game_scores_mean)
    game_scores_min = np.stack(game_scores_min)
    game_scores_max = np.stack(game_scores_max)

    print(game,learners,game_scores_max.max(axis=1))

plt.show()

