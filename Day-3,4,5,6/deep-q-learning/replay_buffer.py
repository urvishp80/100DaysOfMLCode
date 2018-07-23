import random

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, observation, action, reward, next_obs, done):
        data = (observation, action, reward, next_obs, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def _encode_sample(self, indices):
        goals, observations, actions, rewards, next_observations, dones = [], [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            observation, action, reward, next_obs, done = data
            observations.append(np.array(observation, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_observations.append(np.array(next_obs, copy=False))
            dones.append(done)
        return np.array(observations), np.array(actions), np.array(rewards), np.array(next_observations), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        observations: np.array
        actions: np.array
        rewards: np.array
        next_observations: np.array
        dones: np.array
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)