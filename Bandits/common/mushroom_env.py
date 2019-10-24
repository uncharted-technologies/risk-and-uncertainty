import torch
import numpy as np
import pandas as pd

class MushroomEnv:
    """
    Creates a mushroom environment to sample a mushroom and give a reward according the picked action
    """
    def __init__(self):

        data_encoded = pd.read_csv("common/data_mushrooms.csv", index_col=0)

        self.y = torch.from_numpy(data_encoded["target"].values).reshape(-1, 1) * (-2) + 1 # Set y in {-1, 1}
        self.X = torch.from_numpy(data_encoded.drop("target", 1).values)
        self.n_samples = len(data_encoded)
        self.good_mushroom = (1, 1)
        self.bad_mushroom = (-3, 1)
        self.y_hist = []
        self.n_features = self.X.shape[1]
        
    def sample(self, batch_size=1):
        """
        Sample one (X, y)
        """
        if batch_size == 1:
            ix = np.random.randint(0, self.n_samples)
            self.y_sample = self.y[ix].item()
            self.y_hist.append(self.y_sample)
            return self.X[ix, :].unsqueeze(0)
        else:   
            ix = np.random.randint(0, self.n_samples, batch_size)
            self.y_sample = self.y[ix].reshape(-1).numpy()
            self.y_hist += list(self.y_sample)
            return self.X[ix, :]
    
    def clear_y_hist(self):
        self.y_hist = []
    
    
    def hit(self, action):
        if action == 0:
            return 0
        else:
            if self.y_sample == 1:
                return np.random.normal(self.good_mushroom[0], self.good_mushroom[1])
            else:
                return np.random.normal(self.bad_mushroom[0], self.bad_mushroom[1])
    

    def regret(self, action):
        if self.y_sample == 1:
            return self.good_mushroom[0] * (1 - action)
        else:
            return action * self.bad_mushroom[0] * (-1)