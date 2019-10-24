import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pickle
import numpy as np
import matplotlib.pyplot as plt

BBB_NOISE = 0.01
THOMPSON_NOISE = 0.5
EPSILON = 0.05

TO_PLOT = ['thompson','bbb','egreedy']

regrets_thompson = []
regrets_bbb =  []
regrets_egreedy =  []

for subdir, dirs, files in os.walk('results/'):

    if 'Thompson' in subdir and 'thompson' in TO_PLOT:

        info = eval(open(subdir + '/experimental-setup','r').read())

        if info['noise_scale'] == THOMPSON_NOISE:

            log_data_filename = subdir + '/log_data.pkl'
            log_data = pickle.load(open(log_data_filename, 'rb'))
            score_data = np.array(log_data['Cumulative_Regret'])
            scores = score_data[:,0]

            regrets_thompson.append(scores)

    if 'BBB' in subdir and 'bbb' in TO_PLOT:

        info = eval(open(subdir + '/experimental-setup','r').read())

        if info['noise_scale'] == BBB_NOISE:

            log_data_filename = subdir + '/log_data.pkl'
            log_data = pickle.load(open(log_data_filename, 'rb'))
            score_data = np.array(log_data['Cumulative_Regret'])
            scores = score_data[:,0]

            regrets_bbb.append(scores)

    if 'EGreedy' in subdir and 'egreedy' in TO_PLOT:

        info = eval(open(subdir + '/experimental-setup','r').read())

        if info['epsilon'] == EPSILON:

            log_data_filename = subdir + '/log_data.pkl'
            log_data = pickle.load(open(log_data_filename, 'rb'))
            score_data = np.array(log_data['Cumulative_Regret'])
            scores = score_data[:,0]

            regrets_egreedy.append(scores)

regrets_bbb = np.array(regrets_bbb)
regrets_thompson = np.array(regrets_thompson)
regrets_egreedy = np.array(regrets_egreedy)

plt.plot(np.mean(regrets_thompson,axis=0),label = 'Anchoring + our method', color = 'chocolate')
plt.plot(np.mean(regrets_bbb,axis=0),label = 'BBB', color = 'blue')
plt.plot(np.mean(regrets_egreedy,axis=0),label = r'$\epsilon$' + '-greedy 5%', color = 'green')

plt.fill_between(np.arange(20000),
               np.mean(regrets_thompson,axis=0) - 1.96* np.std(regrets_thompson,axis=0)/np.sqrt(regrets_thompson.shape[0]),
               np.mean(regrets_thompson,axis=0) + 1.96* np.std(regrets_thompson,axis=0)/np.sqrt(regrets_thompson.shape[0]),
               facecolor='orange',
               alpha=0.2)

plt.fill_between(np.arange(20000),
               np.mean(regrets_bbb,axis=0) - 1.96* np.std(regrets_bbb,axis=0)/np.sqrt(regrets_bbb.shape[0]),
               np.mean(regrets_bbb,axis=0) + 1.96* np.std(regrets_bbb,axis=0)/np.sqrt(regrets_bbb.shape[0]),
               facecolor='blue',
               alpha=0.2)

plt.fill_between(np.arange(20000),
               np.mean(regrets_egreedy,axis=0) - 1.96* np.std(regrets_egreedy,axis=0)/np.sqrt(regrets_egreedy.shape[0]),
               np.mean(regrets_egreedy,axis=0) + 1.96* np.std(regrets_egreedy,axis=0)/np.sqrt(regrets_egreedy.shape[0]),
               facecolor='green',
               alpha=0.2)

plt.gcf().subplots_adjust(left=0.2,bottom=0.2)
plt.title('Cumulative Regrets', fontsize = 20)
plt.xlabel('Step', fontsize = 18)
plt.ylabel('Regret', fontsize = 18)
plt.xticks([0,10000,20000],fontsize=18)
plt.yticks([0,400,800,1200,1600,2000,2400],fontsize=18)
plt.legend(fontsize = 16)
plt.show()