from typing import List, Union
from turntaking.utils import assert_all_type, merge_dicts
from dataclasses import dataclass
import numpy as np


def index_to_speaker_id(index: int) -> str:
    """
    :param index: number for the speaker
    :return: string
    """
    return chr(ord('a') + index)


def speaker_id_to_index(spk: str) -> int:
    return ord(spk.lower()) - ord('a')


def pluck_speaker_from_samples(samples: List[str], speaker: Union[str, List[str]], keep_original_speaker_name=True):
    if type(speaker) is str:
        speaker = [speaker]

    speaker = sorted(speaker)
    idx = [speaker_id_to_index(spk) for spk in speaker]

    # Select the chosen speakers
    new_samples = activity_to_samples(
        samples_to_activity(samples)[idx]
    )

    # Remap to the original speaker names
    if keep_original_speaker_name:
        letter_map = merge_dicts(
            {chr(ord('a') + new): index_to_speaker_id(old) for new, old in enumerate(idx)},
            {chr(ord('A') + new): index_to_speaker_id(old).upper() for new, old in enumerate(idx)}
        )

        new_samples = [
            ''.join([letter_map[letter] for letter in s])
            for s in new_samples
        ]

    return new_samples


def activity_to_samples(activity_matrix: np.array):
    labels = []
    cols, rows = activity_matrix.shape
    for sample_number in range(rows):
        current_string = ''
        for person_id in range(cols):
            if activity_matrix[person_id, sample_number]:
                current_string += chr(ord('A') + person_id)
            else:
                current_string += chr(ord('a') + person_id)
        labels.append(current_string)

    return labels


def samples_to_activity(samples: List[str]):
    assert len(samples) > 0, 'No labels provided'
    lens = np.array([len(l) for l in samples])
    assert np.all(lens[0] == lens, axis=0), "Expected the same number of labels throughout"
    num_samples = len(samples)
    num_people = len(samples[0])

    matrix = np.zeros((num_people, num_samples), dtype=bool)

    for j, label in enumerate(samples):
        for i, ch in enumerate(label):
            matrix[i, j] = ch.isupper()

    return matrix


def perform_activity_reordering(activity_matrix: np.array):
    """
    Reorder the activity matrix such that the most active speaker is first and the
    least active speaker is last
    :param activity_matrix: (num_spks, num_samples) boolean matrix
    :return:
    """
    totals = np.sum(activity_matrix, axis=1)
    new_ordering = np.argsort(-totals)
    return activity_matrix[new_ordering]


# An utterance that does not belong to any specific session
@dataclass
class Utterance:
    speaker: str
    text: str
    duration: float


@dataclass
class ScenarioUtterance(Utterance):
    start_time: float

    @property
    def end_time(self):
        return self.start_time + self.duration


class Transcript:

    def __init__(self, utterances: List[ScenarioUtterance]):
        # Ensure we are containing utterances at runtime
        assert_all_type(utterances, ScenarioUtterance)
        self.utterances = sorted(utterances, key=lambda utt: utt.end_time)

    @property
    def speakers(self):
        # Sort so we have a consistent ordering
        return sorted(set([utt.speaker for utt in self.utterances]))

    def speaker_activity(self, sample_rate: float, reorder_based_on_activity: bool=True):
        """
        Convert the transcript into a matrix of speaker activity
        :param sample_rate: how often to generate a sample within a second
        :param reorder_based_on_activity: Create a conistent ordering so first speaker is always
        the most active person. Default: True
        :return: 
        """
        number_of_samples = int(self.utterances[-1].end_time * sample_rate)
        number_of_speakers = len(self.speakers)
        speaker_map = dict(zip(self.speakers, range(number_of_speakers)))

        # Matrix will be populated through looping through utterances
        activity_matrix = np.zeros((number_of_speakers, number_of_samples), dtype=bool)

        for utt in self.utterances:
            start_sample = int(utt.start_time * sample_rate)
            end_sample = int(utt.end_time * sample_rate)

            spk_index = speaker_map[utt.speaker]
            activity_matrix[spk_index, start_sample:end_sample+1] = True

        if reorder_based_on_activity:
            return perform_activity_reordering(activity_matrix)
        return activity_matrix

    def samples(self, sample_rate: float, reorder_based_on_activity: bool=True):
        return activity_to_samples(self.speaker_activity(
            sample_rate=sample_rate,
            reorder_based_on_activity=reorder_based_on_activity
        ))


def speakers_in_sample(sample):
    return set(sample.lower())


def active_speakers_in_sample(sample):
    return {letter.lower() for letter in sample if letter.isupper()}


def convert_samples_to_competing(samples, perspective):
    perspective = perspective.lower()
    spks = speakers_in_sample(samples[0])
    other_speakers = spks - {perspective}

    spk_samples = pluck_speaker_from_samples(samples, speaker=perspective)
    other_spk_samples = pluck_speaker_from_samples(samples, speaker=sorted(other_speakers))

    out = []
    for spk_sample, other_sample in zip(spk_samples, other_spk_samples):
        has_competing = len(active_speakers_in_sample(other_sample)) > 0
        competing_str = 'X' if has_competing else 'x'

        # E.g., 'Ax', 'ax', 'AX' etc.
        s = spk_sample + competing_str

        # Special case when it's silent. So all competing models have the
        # same silent state
        if spk_sample.islower() and not has_competing:
            s = 'xx'
        out.append(s)

    return out
