from __future__ import annotations

from typing import List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from turntaking.models.trainers import Trainer
# from turntaking.turntaking_models.markov import silent_state
from turntaking.transcript import active_speakers_in_sample


@dataclass
class Transition:
    previous_state: State
    next_state: State
    weight: float = None


@dataclass
class State(ABC):
    name: str
    transitions: List[Transition] = None

    @property
    def id(self):
        # Mainly used for plotting
        return self.name + '_' + '_'.join([t.next_state.name for t in self.transitions])

    def normalise_weights(self):
        total_out = sum([t.weight for t in self.transitions])
        weights = [t.weight for t in self.transitions]
        diff = max(weights) - min(weights)
        min_weight = min(weights)

        # for t in self.transitions:
        #     if total_out == 0 or diff == 0:
        #         # No count set so just do in uniformally
        #         t.weight = 1/len(self.transitions)
        #         #t.weight = (t.weight - min_weights) / diff
        #     else:
        #         t.weight = (t.weight - min_weight) / diff
        #         #t.weight /= total_out
         
        total_out = sum([t.weight for t in self.transitions])
        for t in self.transitions:
            print('Before: ', t.weight)
            t.weight /= total_out
            print('After: ', t.weight)

    def sample_next_state(self):
        selected_transition: Transition = np.random.choice(
            self.transitions,
            size=1, p=[t.weight for t in self.transitions]
        )[0]
        return selected_transition.next_state


class TransitionModel(ABC):

    def __init__(self, state_names: List[str], trainer=None, smoothing=None):
        assert len(state_names) == len(set(state_names)), "State names must be unique"
        self.states = self.build_states(state_names)
        self.build_connections()
        self.normalise_weights()

        if trainer is None:
            trainer = self.default_trainer()

        self.trainer = trainer
        self.smoothing = smoothing

    def transition_distribution(self, starting_state, samples):
        """
        Estimate the prior prob of transitioning along a transition
        :param starting_state: probably best to use the silent state
        :param samples: How many samples to use for the generation
        :return: prob distribution over the transitions
        """
        dist = self.trainer.transition_distribution(self, starting_state, samples)
        # Make transition order
        return [
            dist[(t.previous_state.name, t.next_state.name)]
            for t in self.transitions
        ]

    def set_weights_from_vector(self, vector):
        assert len(vector) == len(self.transitions), 'Number of params in the model not equal to vector length'
        for transition, weight in zip(self.transitions, vector):
            transition.weight = weight

    @property
    def transitions(self):
        all_trans = []
        for s in self.states:
            all_trans += s.transitions
        return all_trans
        
    @property
    def parameters(self) -> List[float]:
        return [t.weight for t in self.transitions]

    @property
    @abstractmethod
    def default_trainer(self) -> Type[Trainer]:
        pass

    @abstractmethod
    def build_connections(self):
        pass

    @abstractmethod
    def build_states(self, state_names):
        pass

    @abstractmethod
    def sample(self, starting_state: State, samples: int):
        pass

    def find_state_from_data(self, starting_state: str) -> State:
        """
        Find the state in the network, given the data attribute
        """
        for state in self.states:
            if state.name == starting_state:
                return state
        raise ValueError('Cannot find state: ', starting_state)

    def normalise_weights(self):
        for state in self.states:
            state.normalise_weights()

    def fit(self, samples: List[str]):
        self.trainer.fit(samples, model=self)

    def plot(self, dot=None, prune_zero_probs=False, set_notation=False):
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError('Cannot plot the model if Digraph is not installed')

        if dot is None:
            dot = Digraph()

        for node in self.states:
            if set_notation:
                label = '{' +  ', '.join(
                    sorted(active_speakers_in_sample(node.name))
                ) + '}'
                if label == '{}':
                    label = '∅'
            else:
                label = node.name

            label = label.replace('x', 'ξ')

            dot.node(name=node.id, label=label)
            for tran in node.transitions:
                if prune_zero_probs and tran.weight == 0:
                    continue


                dot.edge(
                    tran.previous_state.id,
                    tran.next_state.id,
                    # "{:.2e}".format(tran.weight, 3)
                    str(tran.weight),
                    # constraint='true'
                )

        return dot

"""
Mixins to define different typologies of transition models
"""
# Made for Typing purposes
@dataclass
class HasStates(ABC):
    states: List[State]


class FullyConnectedTypology(ABC):
    def build_connections(self: HasStates):
        for state1 in self.states:
            trans = [Transition(state1, state2, weight=0)
                     for state2 in self.states]
            state1.transitions = trans


class FullyConnectedNoSelfLoopsTypology(ABC):
    def build_connections(self: HasStates):
        for state1 in self.states:
            trans = [Transition(state1, state2, weight=0)
                     for state2 in self.states if state1 != state2]
            state1.transitions = trans
