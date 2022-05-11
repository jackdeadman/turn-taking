import matplotlib.pyplot as plt
from turntaking.stats import Activity as ActivityStats
from turntaking.transcript import samples_to_activity


def plot_activity(activity_matrix, xlim=None):
    if type(activity_matrix[0]) is str:
        activity_matrix = samples_to_activity(activity_matrix)

    plt.imshow(activity_matrix, interpolation='nearest', aspect='auto', cmap='Greys', alpha=0.7)
    if xlim is not None:
        plt.xlim(xlim)


def plot_number_of_people_speaking(activity_matrix):
    if type(activity_matrix[0]) is str:
        activity_matrix = samples_to_activity(activity_matrix)

    distribution = ActivityStats.number_speaking(activity_matrix)
    plt.bar(list(range(len(distribution))), height=distribution)

    plt.title('Number of people speaking in a session')
    plt.xlabel('Number of people speaking.')
    plt.ylabel('Percentage of time speaking.')


def plot_durations_of_people_speaking(activity_matrix, **kwargs):
    if type(activity_matrix[0]) is str:
        activity_matrix = samples_to_activity(activity_matrix)

    num_spks, num_samples = activity_matrix.shape
    spk_durations = {}

    for spk in range(num_spks):
        durations = []
        current_duration = 0
        for i in range(num_samples):
            if activity_matrix[spk, i]:
                current_duration += 1
            elif i != 0 and activity_matrix[spk, i-1]:
                durations.append(current_duration)
                current_duration = 0

        if current_duration != 0:
            durations.append(current_duration)
        spk_durations[spk] = durations

    for spk, durations in spk_durations.items():
        plt.hist(durations, label=spk, alpha=0.5, **kwargs)
    plt.legend()

