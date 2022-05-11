import numpy as np

from domestic_simulation.data.rttm import RTTMSingleFile, RTTMSegment
from turntaking.transcript import samples_to_activity


def convert_samples_to_rttm(name, samples, sample_rate):
    return convert_activity_to_rttm(name, samples_to_activity(samples), sample_rate)


def regions_in_binary_vector(vector):
    vector = list(vector) + [None]
    previous_value = vector[0]
    count = 0
    start_index = 0

    for index, value in enumerate(vector):
        if previous_value == value:
            count += 1
        else:
            end_index = start_index + count
            yield start_index, end_index
            count = 1
            start_index = end_index
            previous_value = value


def convert_activity_to_rttm(name, activity, sample_rate):
    num_speakers, num_samples = activity.shape
    segments = []
    for spk in range(num_speakers):
        for start, end in regions_in_binary_vector(activity[spk]):
            start_time = start/sample_rate
            end_time = end/sample_rate
            duration = end_time - start_time

            assert np.allclose(activity[spk, start:end], activity[spk, start]), 'Region has mixture of 1s and 0???'
            assert activity[spk, start] in [0, 1], 'Binary activity not given'

            if activity[spk, start] == 1:
                segments.append(RTTMSegment(
                    type=None,
                    channel=0,
                    ortho=None,
                    stype=None,
                    conf=None,
                    file=name,
                    begin=start_time,
                    duration=duration,
                    speaker=str(spk))
                )
    return RTTMSingleFile(segments)


def main():
    a = [1,0,0,0,0,1,1,1,0,0,1,0,0,1]
    for start, end in regions_in_binary_vector(a):
        print(a[start:end], start, end)


if __name__ == '__main__':
    main()

