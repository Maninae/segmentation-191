from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

"""
Import this entire file, before importing swish.
"""

def swish_activation(x):
    # Activation as described in https://arxiv.org/pdf/1710.05941.pdf
    # beta = 1 by default
    return x * K.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish_activation)})

