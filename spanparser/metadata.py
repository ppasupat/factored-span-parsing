"""Metadata contains miscellaneous information for training and for
constructing the model (e.g., the vocab should be saved here).
Metadata should be lightweight and serializable.
"""
from __future__ import (absolute_import, division, print_function)
import pickle


DEFAULT_SAVE_KEYS = ['step']


class Metadata(object):

    ################################
    # Generic operations
    
    def __init__(self, config):
        self.save_keys = set(DEFAULT_SAVE_KEYS)
        self.step = 0

    def save(self, filename):
        to_save = {k: getattr(self, k) for k in self.save_keys}
        with open(filename, 'wb') as fout:
            pickle.dump(to_save, fout)

    def load(self, filename):
        with open(filename, 'rb') as fin:
            loaded = pickle.load(fin)
        for k, v in loaded.items():
            setattr(self, k, v)

    ################################
    # Key-specific operations

