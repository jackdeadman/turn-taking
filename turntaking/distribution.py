from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import lognorm, beta
import scipy.stats


class Distribution(ABC):

    @property
    @abstractmethod
    def parameters(self):
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, params):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def pdf(self, x):
        pass

    @property
    @abstractmethod
    def has_been_fit(self):
        pass

    def expected_value(self, x):
        return np.sum(x * self.pdf(x))


class ScipyDistribution(Distribution, ABC):

    @property
    def parameters(self):
        return self.__params

    @parameters.setter
    def parameters(self, params):
        self.__params = params

    @property
    @abstractmethod
    def model(self):
        pass

    def __init__(self):
        self.__params = None

    def fit(self, data):
        if len(data):
            self.__params = self.model.fit(data, loc=0)

    def sample(self):
        assert self.has_been_fit, 'Need to fit on data before sampling'
        distribution_params = self.__params[:-2]
        loc, scale = self.__params[-2:]
        time = -1
        attempts = 0
        max_attempts = 200

        while time <= 1:
            time = self.model.rvs(*distribution_params, loc=loc, scale=scale, size=1)
            if attempts > max_attempts:
                raise ValueError('Took too many attempts to sample a positive value. The distribution has probably been fit on negative data.')
            attempts += 1
        return time

    def pdf(self, x):
        distribution_params = self.__params[:-2]
        loc, scale = self.__params[-2:]
        return self.model.pdf(x, *distribution_params, loc=loc, scale=scale)

    @property
    def has_been_fit(self):
        return self.__params is not None


class LogNormalDistribution(ScipyDistribution):
    @property
    def model(self):
        return lognorm


class BetaDistribution(ScipyDistribution):
    @property
    def model(self):
        return beta


class ScipyDistributionGenerator(ScipyDistribution):

    def __init__(self, model_name):
        super().__init__()
        self.__model = getattr(scipy.stats, model_name)

    @property
    def model(self):
        return self.__model
