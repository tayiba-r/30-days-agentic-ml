import numpy as np

#This models a k-armed bandit where each arm has an unknown mean reward. 
#Pulling an arm gives you a noisy reward centered around that arm’s true mean.

class Bandit:
    def __init__(self, k=10, seed=None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.true_means = self.rng.normal(0,1,size=k)

    def pull(self, arm):
        reward = self.rng.normal(self.true_means[arm],1.0)
        return reward