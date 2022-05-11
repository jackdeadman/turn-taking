from itertools import chain, combinations
from typing import List
from abc import ABCMeta
import dill as pickle
import numpy as np
import yaml
from numpy.lib.stride_tricks import as_strided


class Timer:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def samples(self, *, seconds=0, minutes=0, hours=0):
        """
        :param seconds:
        :param minutes:
        :param hours:
        :return: The specified time in samples

        >>> time = Timer(8000)
        >>> time.samples(minutes=1)
        480000
        """
        s = seconds * self.sample_rate
        m = minutes * 60 * self.sample_rate
        h = hours * 60**2 * self.sample_rate
        return s + m + h


def assert_all_type(list_of_objects: List[object], datatype: type):
    for obj in list_of_objects:
        # TODO: Using a hack for now as idk why checking the type directly isn't working...
        # I think it's to do with Python imports
        assert str(type(obj)) == str(datatype), 'Expected ' + str(datatype)\
                                          + ' but got instance of ' + str(type(obj))


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def sort_string(string, key):
    return ''.join(sorted(string, key=key))


def align_yaxis(ax1, v1, ax2, v2):
    """
    https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


def chunk_list(iterable, chunk_size, include_last=True):
    current_size = 0
    current_chunk = []

    for element in iterable:
        current_size += 1
        current_chunk.append(element)
        if current_size == chunk_size:
            yield current_chunk
            current_chunk = []
            current_size = 0

    if include_last and len(current_chunk):
        yield current_chunk


def frame(x, *, frame_length, hop_length, axis=-1, writeable=False, subok=False):
    """
    Implementation derived from: http://librosa.org/doc/main/_modules/librosa/util/utils.html#frame
    """
    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ValueError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise ValueError("Invalid hop_length: {:d}".format(hop_length))

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def flatten_list(l):
    new_list = []
    for element in l:
        if type(element) is list:
            for inner_element in element:
                new_list.append(inner_element)
        else:
            new_list.append(element)
    return new_list


def load_pickle(data: str):
    with open(data, 'rb') as f:
        return pickle.load(f)


def write_pickle(path: str, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_yaml(path: str):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(path, data):
    with open(path, 'w') as f:
        return yaml.dump(data, f)


def merge_dicts(d1, d2):
    return {**d1, **d2}


def num_to_letter(num, upper=False):
    letter = chr(ord('a') + num)
    if upper:
        letter = letter.upper()
    return letter

