from collections.abc import Mapping
import numpy as np
from copy import deepcopy
from .seeding import np_random


class ParameterDistribution(Mapping):
    """
    Defines a list of parameters with associated distributions.
    Can be sampled, reset to nominal values.
    """

    def __init__(self, *args, **kwargs):
        self.current = {}
        self.nominal = {}
        self.ranges = {}

        self.np_random = np.random.RandomState()

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return seed

    def register(self, name, nominal, range=None):
        self.current[name] = nominal
        self.nominal[name] = nominal
        self.ranges[name] = range

    def reset(self):
        for k,v in self.nominal.items():
            self.current[k] = v

    def sample(self):
        for k, rng in self.ranges.items():
            if rng is None:
                self.current[k] = self.nominal[k]
            else:
                self.current[k] = self.np_random.uniform(low=rng[0], high=rng[1])

    def values(self):
        """
        returns copy of current values
        """
        return deepcopy(self.current)

    def __getitem__(self, key):
        return self.current[key]

    def __iter__(self):
        return iter(self.current)

    def __repr__(self):
        return repr(self.current)

    def __len__(self):
        return len(self.current)
