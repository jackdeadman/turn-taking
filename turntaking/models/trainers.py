from __future__ import annotations

from typing import List
from abc import ABC, abstractmethod
from collections import Counter, defaultdict


class Trainer(ABC):

    @abstractmethod
    def fit(self, samples: List[str], model: TransitionModel):
        pass


class SimpleTrainer(Trainer):

    def __init__(self, smoothing=0):
        self.smoothing = smoothing

    def statistics(self, samples):
        state_transition_counter = Counter()

        for prev, current in zip(samples, samples[1:]):
            state_transition_counter[(prev, current)] += 1

        return state_transition_counter

    def transition_distribution(self, model: MarkovModel, starting_state, num_samples):
        samples = model.sample(starting_state, num_samples)
        stats = self.statistics(samples)
        total = sum(stats.values())
        return Counter({key: value / total for key, value in stats.items()})

    def fit(self, samples, model: MarkovModel):
        state_stats = self.statistics(samples)

        for state in model.states:
            for transition in state.transitions:
                count = state_stats[(transition.previous_state.name, transition.next_state.name)]
                transition.weight = count + self.smoothing
        model.normalise_weights()


class SimpleSemiMarkovTrainer(SimpleTrainer):

    def statistics(self, samples):
        assert len(samples) != 0, 'Need some samples to train'
        durations = defaultdict(list)
        count = 0

        for prev, current in zip(samples, samples[1:]):
            if prev == current:
                count += 1
            else:
                durations[(prev, current)].append(count)
                count = 0

        return durations

    def transition_distribution(self, model: MarkovModel, starting_state, num_samples):
        samples = model.sample(starting_state, num_samples)
        stats = self.statistics(samples)
        total = sum([len(counts) for counts in stats.values()])

        tran_dist = Counter({key: len(value) / total for key, value in stats.items()})

        state_dict = Counter()

        for (prev, next_state), counts in stats.items():
            state_dict[next_state] += len(counts) / total

        return tran_dist, state_dict

    def fit(self, samples, model: SemiMarkovModel):
        durations_map = self.statistics(samples)

        for state in model.states:
            all_durations = []
            for transition in state.transitions:
                durations = durations_map[(transition.previous_state.name, transition.next_state.name)]
                all_durations += durations
                count = len(durations) + self.smoothing
                transition.weight = count

            state.normalise_weights()
            #
            # for t in state.transitions:
            #     if t.next_state == state:
            #         if t.weight != 0:
            #             print('Counts for this transition: ', len(durations_map[(
            #                 t.previous_state.name,
            #                 t.next_state.name
            #             )]))
            #         assert t.weight == 0, 'Self-loops should be zero-weighted.'

            state.fit(all_durations)


class SharedWeightsSemiMarkovTrainer(SimpleSemiMarkovTrainer):

    def fit(self, samples, model):
        super().fit(samples, model)
        #type(model.states[0]).merge_states()
