from __future__ import annotations

from dataclasses import dataclass
from abc import ABC

import numpy as np

from turntaking.models.base import Transition, TransitionModel, State, FullyConnectedTypology
from turntaking.models.trainers import SimpleTrainer


class MarkovModel(TransitionModel, ABC):

    @property
    def default_trainer(self):
        return SimpleTrainer

    def build_states(self, state_names):
        return [MarkovState(name) for name in state_names]

    def sample(self, starting_state: str, samples: int):
        state = self.find_state_from_data(starting_state)
        generated_samples = []
        from tqdm import tqdm

        for i in tqdm(range(samples)):
            generated_samples.append(state.name)
            state = state.sample_next_state()

        return generated_samples


@dataclass
class MarkovState(State):
    pass


class FullyConnectedMarkovModel(FullyConnectedTypology, MarkovModel):
    pass
