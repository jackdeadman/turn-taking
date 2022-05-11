import numpy as np

# Create a namespace on Activity as all these stats relate to the activity matrix
class Activity:

    @staticmethod
    def number_speaking(activity_matrix: np.array):
        amount_speaking_per_frame = activity_matrix.sum(axis=0)
        _, counts = np.unique(amount_speaking_per_frame, return_counts=True)
        #assert len(counts) == activity_matrix.shape[0] + 1, 'Expected a count for every person.'

        distribution = counts / np.sum(counts)
        return distribution