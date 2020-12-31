import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from copy import deepcopy
from common.utils import RunningMeanStd
import gym

class Env():

    def __init__(self, normalize):
        #shared parameters
        self.env = gym.make('Pendulum-v0')
        self.T = 200
        self.max_steps = 200
        self.n_signal = 4
        self.n_agent = 1
        self.n_actions = self.env.action_space.shape[0]
        self.n_episode = 1000
        self.max_u = None
        self.n_neighbors = 2
        self.input_size = self.env.observation_space.shape[0]
        self.nD = self.n_agent
        self.GAMMA = 0.9

        self.fileresults = open('learning.data', "w")
        self.normalize = normalize
        self.compute_neighbors = False
        self.neighbors_size = 2 #max number of neighbor
        self.compute_neighbors_last = np.array([[0], [1]])
        self.compute_neighbors_last_index = [list(range(len(self.compute_neighbors_last[i]))) for i in range(self.n_agent)]
        if normalize:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]

    def __del__(self):
        self.fileresults.close()

    def toggle_compute_neighbors(self):
        pass
    
    def neighbors(self):
        return (self.compute_neighbors_last, self.compute_neighbors_last_index)

    def reset(self):
        self.rinfo = np.array([0.] * self.n_agent)
        obs = self.env.reset()
        return self.parse_obs(obs)

    def parse_obs(self, obs):
        h = []
        for k in range(self.n_agent):
            h.append(obs)

        if self.normalize:
            for i in range(self.n_agent):
                h[i] = list(self.obs_rms[i].obs_filter(np.array(h[i])))

        return h

    def step(self, action):
        # action is in [0;1]
        action = action[0] * 4 - 2
        obs, re, done, info = self.env.step(action)

        self.rinfo += re
        return self.parse_obs(obs), [re], done

    def end_episode(self):
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()

    def render(self):
        self.env.render()
