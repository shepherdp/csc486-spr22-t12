"""

"""

import numpy as np
import random
from scipy.spatial import distance

POS = ['Y', 'y', 'Yes', 'yes', 'YES']
NEG = ['N', 'n', 'No', 'no', 'NO']
TRU = ['T', 't', 'True', 'true', 'TRUE', True]
FLS = ['F', 'f', 'False', 'false', 'FALSE', False]


def coin_flip(prob):
    """
    Simple weighted coin flip.  Return True with probability = param:prob
    :param prob: Probability to return True
    :return:
    """

    if random.random() < prob:
        return True
    return False


def convert(val, t):
    """
    Convert param:val to type param:t
    :param val:
    :param t:
    :return:
    """
    if t == 'int':
        return int(val)
    if t == 'float':
        return float(val)
    if t == 'bool':
        return True if val in TRU else False
    if t == 'string':
        return val


def dist(vec1, vec2):
    """
    Extract visible dimensions from param:vec2 and compare them to param:vec1.
    :param vec1: perceiving agent's feature vector
    :param vec2: perceived agent's feature vector
    :return: Hamming distance between visible dimensions of param:vec1 and param:vec2
    """
    distvec1 = [vec1[i] for i in range(len(vec1)) if vec2[i] != 0.]
    if not distvec1:
        return 0.
    distvec2 = [vec2[i] for i in range(len(vec2)) if vec2[i] != 0.]
    return distance.hamming(distvec1, distvec2)


def is_type(val, t):
    """
    Determine if param:val is of type param:t.
    :param val: the value to check
    :param t: a string representing the type to check against
    :return: True if param:val is of type param:t, False otherwise.
    """
    if t == 'int':
        try:
            int(val)
            return True
        except ValueError:
            return False
    elif t == 'float':
        try:
            float(val)
            return True
        except ValueError:
            return False
    elif t == 'bool':
        if val in TRU + FLS:
            return True
        else:
            return False
    elif t == 'string':
        return True


def attr_str(vec):
    """
    Produce a string representation of the iterable of integers param:vec.
    :param vec: the list of integers to stringify
    :return: string representation of param:vec.
    """
    string = ''
    for i in vec:
        string += str(int(i))
    return string


def matrix_to_string(m):
    """
    Encode 2D matrix param:m as string with # delimiting comma-delimited rows.
    Matrices encoded with this function can be decoded back into matrices with
    string_to_matrix().
    :param m: nested list
    :return: string representation of param:m
    """
    s = ''
    for row in m:
        for elem in row:
            s += f'{elem},'
        s = s[:-1] + '#'
    return s[:-1]


def string_to_matrix(s):
    """

    :param s:
    :return:
    """
    ret = []
    split = s.split('#')
    for line in split:
        splitline = string_to_vector(line)
        ret.append(splitline)
    return np.array(ret)


# Encode vector as comma-delimited string
def vector_to_string(v):
    """

    :param v:
    :return:
    """
    s = ''
    for elem in v:
        s += '%s,' % elem
    return s[:-1]


def string_to_vector(s):
    """

    :param s:
    :return:
    """
    split = s.split(',')
    if split == ['']:
        return []
    return [float(i) for i in split]


# Special encoding for reading/writing mask matrices.
# Since rows are binary, just encode them as base 10 numbers to save
# space.
def encode_mask(masks):
    """

    :param masks:
    :return:
    """
    s = ''
    for i in range(len(masks)):
        if not any( masks[i] ): continue
        s += str(i) + ','
        temp = ''
        for j in range( len( masks[i] ) ):
            if masks[i][j] == 0.0:
                temp += '0'
            else:
                temp += '1'
        s += '%d#' % int(temp, 2)
    return s[:-1]


# Take encoded mask entries
def decode_mask(mask):
    """

    :param mask:
    :return:
    """
    return [int(i) for i in str(bin(int(mask)))[2:]]


def authorize_overwrite(name):
    """

    :param name:
    :return:
    """
    print(f'About to overwrite: {name}')
    choice = ''
    while choice not in POS + NEG:
        choice = input('Continue? (y/n) ')
    if choice in POS:
        return True
    if choice in NEG:
        return False
