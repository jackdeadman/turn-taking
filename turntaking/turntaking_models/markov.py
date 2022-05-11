# Combines simple models into full turntaking
import matplotlib.pyplot as plt
import numpy as np

from turntaking.models.base import Transition
from turntaking.models.semi_markov import FullyConnectedSemiMarkovModel, state_types, SemiMarkovModel
from turntaking.transcript import pluck_speaker_from_samples, index_to_speaker_id, convert_samples_to_competing
from turntaking.models.markov import MarkovModel
from typing import List, Dict, Iterable, Type
from turntaking.models.markov import FullyConnectedMarkovModel
from turntaking.utils import powerset, sort_string


class IndependentTurntakingModel:

    """
    Wraps models together and runs them simultaneously
    """

    # TODO: the type here might need to be more generic. I.e. transition model
    def __init__(self, models: Dict[str, SemiMarkovModel]):
        self.models = models

    @property
    def speakers(self):
        return list(self.models)

    def fit(self, samples):
        for spk, model in self.models.items():
            training_samples = pluck_speaker_from_samples(samples, spk)
            model.fit(training_samples)

    def sample(self, starting_state: str, samples: int):
        """
        :param starting_state: AbCd
        :param samples:
        :return:
        """

        streams = []

        for starting_state, model in zip(starting_state, self.models.values()):
            streams.append(model.sample(starting_state, samples))

        streams = np.array(streams)

        return [
            ''.join(streams[:, i])
            for i in range(samples)
        ]

    def plot(self, dot=None, **args):
        for model in list(self.models.values())[::-1]:
            dot = model.plot(dot=dot, **args)
        return dot

    @property
    def parameters(self):
        params = []
        for model in self.models.values():
            params += model.parameters
        return params

    @property
    def states(self):
        states = []
        for model in self.models.values():
            states += model.states
        return states

    @property
    def steady_state_parameters(self):
        params = []
        for model in self.models.values():
            params += model.steady_state_parameters
        return params

    @property
    def expected_durations(self):
        params = []
        for model in self.models.values():
            params += model.expected_durations
        return params

    @property
    def expected_overlap(self):
        params = []
        for model in self.models.values():
            params += [model.expected_overlap]
        return params

    @property
    def steady_state_transition_probs(self):
        params = []
        for model in self.models.values():
            params += model.steady_state_transition_probs
        return params


    def normalise_weights(self):
        for model in self.models.values():
            model.normalise_weights()


class FullModelWrapper:

    def __init__(self, model: Type[FullyConnectedMarkovModel], num_spks, model_args):
        states = full_speaker_states(num_spks)
        self.model = model(states, **model_args)
        self.num_spks = num_spks

    @property
    def silent_state(self):
        return silent_state(self.num_spks)

    def fit(self, samples):
        self.model.fit(samples)

    def plot(self, *args, **kargs):
        return self.model.plot(*args, **kargs)

    def sample(self, *args, **kargs):
        return self.model.sample(*args, **kargs)

    def set_weights_from_vector(self, vector):
        self.model.set_weights_from_vector(vector)

    @property
    def states(self):
        return self.model.states

    @states.setter
    def states(self, states):
        self.states = states

    @property
    def parameters(self):
        return self.model.parameters

    def normalise_weights(self):
        self.model.normalise_weights()

    def transition_distribution(self, *args, **kwargs):
        return self.model.transition_distribution(*args, **kwargs)

    @property
    def steady_state_parameters(self):
        return self.model.steady_state_parameters


class FullModelSemiMarkovWrapper(FullModelWrapper):

    def __init__(self, model: Type[FullyConnectedSemiMarkovModel], num_spks, model_args, state_type):
        states = full_speaker_states(num_spks)
        self.model = model(states, state_type=state_type, **model_args)
        self.num_spks = num_spks

    def plot_states(self, **args):
        return self.model.plot_states(**args)

    @property
    def model_params(self):
        return self.model.model_params

    @property
    def expected_overlap(self):
        return self.model.expected_overlap

    @property
    def expected_durations(self):
        return self.model.expected_durations

    @property
    def steady_state_transition_probs(self):
        return self.model.steady_state_transition_probs




class IndependentCompetingTurntakingModel(IndependentTurntakingModel):

    def fit(self, samples):
        for spk, model in self.models.items():
            training_samples = convert_samples_to_competing(samples, perspective=spk)
            model.fit(training_samples)

    def plot_states(self, *args, **kwargs):
        for i, model in enumerate(self.models.values()):
            plt.figure()
            model.plot_states(*args, **kwargs)

    @property
    def parameters(self):
        models = list(self.models.values())
        params = models[0].parameters

        for model in models[1:]:
            # TODO: Fix this. This is assuming the silent state params are the first two in the model
            # print(model.parameters[8:10])
            p = model.parameters[::]
            del p[:8:10]
            params += p
        return params



def number_of_people_speaking(sample: str):
    return {letter for letter in sample if letter.isupper()}


def number_of_changes(transition: Transition):
    speaking_before = number_of_people_speaking(transition.previous_state.name)
    speaking_after = number_of_people_speaking(transition.next_state.name)

    started_speaking = speaking_after - speaking_before
    stopped_speaking = speaking_before - speaking_after

    return len(started_speaking) + len(stopped_speaking)


def prune_model(model):
    for state in model.states:
        state.transitions = [
            t
            for t in state.transitions if number_of_changes(t) <= 1
        ]
    model.normalise_weights()


def expand_independent_semi_markov(model: IndependentTurntakingModel):
    # states = full_speaker_states(len(model.models))
    print(model.models['a'].state_type)
    new_model = TurntakingFactory.full_semi_markov(
        len(model.models),
        state_type=model.models['a'].state_type
    )

    state_map = {}
    tran_map = {}

    for spk, spk_model in model.models.items():
        for state in spk_model.states:
            state_map[(spk, state.name)] = state
            for t in state.transitions:
                tran_map[(spk, state.name, t.next_state.name)] = t

    # Train on all the data in the relevant states
    for new_state in new_model.states:
        all_data_trained_on = []
        for spk in new_state.name:
            spk_id = spk.lower
            all_data_trained_on += state_map[(spk_id, spk)]
        new_state.fit(all_data_trained_on)

    # Compute the transition probs
    for new_state in new_model.states:
        for tran in new_state.transitions:
            for spk in new_state.name:
                old_tran = tran_map[(spk, spk, )]


    return new_model


def speaker_ids(num_spks: int) -> List[str]:
    """
    Get speaker ids
    :param num_spks:
    :return:
    """
    return list([index_to_speaker_id(i) for i in range(num_spks)])


def speaking_to_state_string(people_speaking: Iterable[str], num_spks: int):
    """
    :param people_speaking: iterable of people who are speaking. Lowercase strings of their ids
    :param num_spks: number of people that could be speaking
    :return:
    """
    # e.g., {'a','b','c','d'}
    all_speakers = set(speaker_ids(num_spks))
    # e.g., {'b', 'c'}
    speaking = set(people_speaking)
    # e.g., {'a', 'd'}
    not_speaking = all_speakers - speaking

    speaking_string = ''.join(speaking).upper()
    not_speaking_string = ''.join(not_speaking)

    return sort_string(
        speaking_string + not_speaking_string,
        key=lambda s: s.lower()
    )


def silent_state(num_spks):
    return speaking_to_state_string(people_speaking=[], num_spks=num_spks)


def silent_state_competing():
    return 'xx'


def full_speaker_states(num_spks):
    spk_ids = speaker_ids(num_spks)

    return [
        speaking_to_state_string(people_speaking, num_spks=num_spks)
        for people_speaking in powerset(spk_ids)
    ]


class TurntakingFactory:
    """Using the class as a namespace"""

    @staticmethod
    def independent_semi_markov(num_spks, state_type, **args) -> IndependentTurntakingModel:

        if type(state_type) is str:
            state_type = state_types.scipy(state_type)

        spks = speaker_ids(num_spks)
        model = IndependentTurntakingModel(models={
            spk: FullyConnectedSemiMarkovModel(
                state_type=state_type,
                state_names=[spk.lower(), spk.upper()], **args)
            for spk in spks}
        )
        return model

    @staticmethod
    def independent(num_spks, **args) -> IndependentTurntakingModel:
        spks = speaker_ids(num_spks)
        model = IndependentTurntakingModel(
            models={
            spk: FullyConnectedMarkovModel(state_names=[
                spk.lower(), spk.upper()
            ], **args) for spk in spks}
        )
        return model

    @staticmethod
    def competing(num_spks, **args):
        spks = speaker_ids(num_spks)
        model = IndependentCompetingTurntakingModel(models={
            spk: FullyConnectedMarkovModel(state_names=[
                'xx', spk.lower() + 'X', spk.upper() + 'x', spk.upper() + 'X'
            ], **args) for spk in spks
        })
        return model

    @staticmethod
    def competing_semi_markov(num_spks, state_type, **args):
        if type(state_type) is str:
            state_type = state_types.scipy(state_type)
        spks = speaker_ids(num_spks)
        model = IndependentCompetingTurntakingModel(models={
            spk: FullyConnectedSemiMarkovModel(state_type=state_type, state_names=[
                'xx', spk.lower() + 'X', spk.upper() + 'x', spk.upper() + 'X'
            ], **args) for spk in spks
        })
        return model

    @staticmethod
    def full(num_spks, **args) -> FullyConnectedMarkovModel:
        states = full_speaker_states(num_spks)
        return FullModelWrapper(model=FullyConnectedMarkovModel, num_spks=num_spks, model_args=args)
        # return FullyConnectedMarkovModel(state_names=states, **args)

    @staticmethod
    def full_semi_markov(num_spks, state_type, **args) -> FullModelSemiMarkovWrapper:
        if type(state_type) is str:
            state_type = state_types.scipy(state_type)

        return FullModelSemiMarkovWrapper(
            model=FullyConnectedSemiMarkovModel,
            num_spks=num_spks,
            state_type=state_type,
            model_args=args
        )

def main():
    num_spks = 4
    spks = list([chr(ord('a') + i) for i in range(num_spks)])

    model = IndependentTurntakingModel(models={
        spk: FullyConnectedMarkovModel(state_names=[
            spk.lower(), spk.upper()
        ]) for spk in spks
    })


if __name__ == '__main__':
    main()
