from __future__ import annotations
from typing import Type, Union

from abc import abstractmethod, ABC
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from turntaking.distribution import Distribution, LogNormalDistribution, BetaDistribution, ScipyDistributionGenerator
from turntaking.models.base import TransitionModel, State, FullyConnectedTypology, FullyConnectedNoSelfLoopsTypology
from turntaking.models.trainers import Trainer, SimpleSemiMarkovTrainer
from turntaking.transcript import active_speakers_in_sample
from turntaking.utils import align_yaxis, flatten_list

@dataclass
class SemiMarkovState(State):
    data_trained_on = None
    __model = None

    # @abstractclassmethod
    # def merge_states(cls, new_name: str, s1: SemiMarkovState, s2: SemiMarkovState):
    #     assert cls is not SemiMarkovState, 'Must be a derivative concrete class'
    #     new_state = cls(name=new_name)
    #     all_data = s1.data_trained_on + s2.data_trained_on
    #     new_state.fit(all_data)

    @property
    def parameters(self):
        return self.__model.parameters

    @parameters.setter
    def parameters(self, params):
        self.__model.parameters = params

    def expected_value(self, x):
        return self.__model.expected_value(x)

    def normalise_weights(self):

        total = sum([t.weight for t in self.transitions])

        for t in self.transitions:
            if total == 0:
                t.weight = 1/len(self.transitions)
            else:
                t.weight /= total

        return

        #total_out = sum([
        #    t.weight for t in self.transitions
        #    if t.previous_state != t.next_state]
        #)
        weights = [t.weight for t in self.transitions]
        total_out = sum(weights)
        diff = max(weights) - min(weights)
        min_weight = min(weights)

        has_self_loop = any(t.weight for t in self.transitions
                            if t.previous_state == t.next_state)

        if has_self_loop:
            num_trans = len(self.transitions) - 1
        else:
            num_trans = len(self.transitions)

        for t in self.transitions:
            if t.previous_state == t.next_state:
                t.weight = 0
                continue

            if total_out == 0 or diff == 0:
                # No count set so just do in uniformally
                t.weight = 1/num_trans
            else:
                #t.weight /= total_out
                t.weight = (t.weight - min_weight) / diff

        total = sum([t.weight for t in self.transitions])
        for t in self.transitions:
            t.weight /= total


    @property
    @abstractmethod
    def distribution_constructor(self) -> Type[Distribution]:
        pass

    def sample(self):
        # Ensure it is an int as we are handling samples
        return round(float(self.__model.sample()))

    def fit(self, training_data):
        self.data_trained_on = training_data
        self.__model = self.distribution_constructor()
        self.__model.fit(training_data)

    def pdf(self, x):
        """
        :param x: vector of observations
        :return: p(x)
        """
        return self.__model.pdf(x)

    @property
    def likelihood(self):
        return np.sum(np.log(self.__model.pdf(self.data_trained_on)))

    def plot(self, include_training_data=True, bins=None, **args):
        # Cannot plot if it has not data.
        if self.data_trained_on is None or len(self.data_trained_on) == 0:
            return
        ax = plt.gca()
        original_axis = ax

        if include_training_data:
            ax.hist(self.data_trained_on, bins=bins, alpha=0.5)
            ax = ax.twinx()

        x = np.arange(-max(self.data_trained_on), max(self.data_trained_on))
        ax.plot(x, self.pdf(x), c='y', **args)

        if include_training_data:
            # If we are using two axes, make sure they both line up at y=0
            align_yaxis(original_axis, 0, ax, 0)


class SemiMarkovModel(TransitionModel, ABC):

    def __init__(self, state_names, trainer=None, state_type=None, smoothing=None):
        if state_type is None:
            state_type = LogNormalSemimarkovState
        self.state_type = state_type

        super().__init__(state_names=state_names, trainer=trainer, smoothing=smoothing)

    def transition_distribution(self, starting_state, samples):
        dist, state_dist = self.trainer.transition_distribution(self, starting_state, samples)
        # Make transition order
        tran_dist = [
            dist[(t.previous_state.name, t.next_state.name)]
            for t in self.transitions
        ]

        state_dist = [state_dist[s.name] for s in self.states]
        return tran_dist, state_dist

    def set_weights_from_vector(self, vector):
        """
        :param vector: format is,
        transition_weights + [ state1_parameters_1, state1_parameters_2, state2_parameters_1, state2_parameters_2 ...]
        :return:
        """
        transition_weights = vector[:len(self.transitions)]
        state_weights = vector[len(self.transitions):]

        super().set_weights_from_vector(transition_weights)

        for i, state in enumerate(self.states):
            num_params = len(state.parameters)
            start_position = i*num_params
            state_params = state_weights[start_position:start_position + num_params]
            state.parameters = state_params

    @property
    def model_params(self):
        exp_durations = []
        for s in self.states:
            params = s.parameters

            if params is None:
                exp_durations.append(0)
            else:
                low = np.min(s.data_trained_on)
                high = np.max(s.data_trained_on)
                exp_durations.append(s.expected_value(np.linspace(low, high, 10000)))
        return exp_durations


    @property
    def expected_durations(self):
        values = []

        for s, prob in zip(self.states, self.steady_state):
            spks = active_speakers_in_sample(s.name)

            if len(s.data_trained_on):
                low = np.min(s.data_trained_on)
                high = np.max(s.data_trained_on)
                dur = s.expected_value(np.linspace(low, high, 10000))
            else:
                dur = 0

            dur *= prob
            values.append(dur)

        print('values: ', values)
        return values


    @property
    def expected_overlap(self):
        top_values = []
        bottom_values = []
        for s, prob in zip(self.states, self.steady_state):
            spks = active_speakers_in_sample(s.name)
            if len(spks) == 0:
                continue

            if len(s.data_trained_on):
                low = np.min(s.data_trained_on)
                high = np.max(s.data_trained_on)
                dur = s.expected_value(np.linspace(low, high, 10000))
            else:
                dur = 0

            dur *= prob

            if len(spks) > 1:
                top_values.append(dur)

            bottom_values.append(dur)

        print('COMPTUING: ', top_values, bottom_values, sum(top_values) / sum(bottom_values))
        return sum(top_values) / sum(bottom_values)



    @property
    def model_params_old(self):
        model_params = []
        num_model_params = 2

        for s in self.states:
            params = s.parameters
            if params is None:
                params = [None]
            model_params += params


        # Replace any Nones with 0s the size of the distribution params
        for i, p in enumerate(model_params):
            if p is None:
                p = [0] * num_model_params
                model_params[i] = p

        return flatten_list(model_params)

    @property
    def transitions(self):
        return [t for t in super().transitions if t.previous_state != t.next_state]

    @property
    def transition_params(self):
        # Exclude self-transitions
        transition_weights = [t.weight for t in self.transitions
                              if t.previous_state != t.next_state]
        return transition_weights

    @property
    def parameters(self) -> List[float]:
        return self.transition_params + self.model_params

    @property
    def steady_state(self):
        t = self.transition_matrix
        dim = t.shape[0]
        q = (t-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)

    def transition_matrix_to_realised_vector(self, transition_matrix):
        output = []
        state_index = {state.name: i for i, state in enumerate(self.states)}
        for s1 in self.states:
            index1 = state_index[s1.name]
            for tran in s1.transitions:
                index2 = state_index[tran.next_state.name]
                output.append(transition_matrix[index1, index2])
        return output

    @property
    def steady_state_transition_probs(self):
        transition_matrix = self.transition_matrix
        ss = self.steady_state
        num_states = len(transition_matrix)

        output = np.zeros_like(transition_matrix)

        for i in range(num_states):
            for j in range(num_states):
                output[i, j] = ss[i] * transition_matrix[i, j]

        return self.transition_matrix_to_realised_vector(output)

    @property
    def transition_matrix(self):
        state_index = {state.name: i for i, state in enumerate(self.states)}
        transitions = np.zeros((len(state_index), len(state_index)))

        for s1 in self.states:
            index1 = state_index[s1.name]
            for tran in s1.transitions:
                index2 = state_index[tran.next_state.name]
                transitions[index1, index2] = tran.weight
        return transitions

    @property
    def steady_state_parameters(self):
        return list(self.steady_state_transition_probs) + list(self.steady_state)

    @property
    def default_trainer(self) -> Type[Trainer]:
        return SimpleSemiMarkovTrainer

    def build_states(self, state_names):
        return [self.state_type(name) for name in state_names]

    def sample(self, starting_state: Union[str, SemiMarkovState], samples: int):
        """
        :param starting_state: string or SemiMarkovState
        :param samples: number of samples to generate
        :return: List of strings of state names
        """
        if type(starting_state) is str:
            starting_state = self.find_state_from_data(starting_state)

        current_state = starting_state
        samples_generated = 0
        output = []

        while samples_generated <= samples:
            d = starting_state.sample()
            output += d * [current_state.name]
            current_state = current_state.sample_next_state()
            samples_generated += d

        return output[:samples]

    def plot_states(self, include_training_data=True, **args):
        for i, state in enumerate(self.states):
            plt.subplot(4, len(self.states) // 4 + 1, i+1)
            plt.title(state.name)
            state.plot(include_training_data, **args)

    # def plot(self, **args):
    #     if plot_args is None:
    #         plot_args = {}
    #     #self.plot_states(include_training_data=include_training_data, **args)
    #     return super().plot(**plot_args)

@dataclass
class LogNormalSemimarkovState(SemiMarkovState):

    @property
    def distribution_constructor(self) -> Type[Distribution]:
        return LogNormalDistribution

@dataclass
class BetaSemimarkovState(SemiMarkovState):

    @property
    def distribution_constructor(self) -> Type[Distribution]:
        return BetaDistribution


@dataclass
class ScipyState(SemiMarkovState):
    dist_name: str = 'lognorm'

    @property
    def distribution_constructor(self) -> Type[Distribution]:
        return lambda: ScipyDistributionGenerator(self.dist_name)



class FullyConnectedSemiMarkovModel(FullyConnectedNoSelfLoopsTypology, SemiMarkovModel):
    pass


# Using as a namespace.
# Used for convenience
class state_types:
    lognorm = LogNormalSemimarkovState
    beta = BetaSemimarkovState
    scipy = lambda dist_name: lambda *args, **kwargs: ScipyState(
        *args, dist_name=dist_name, **kwargs)
