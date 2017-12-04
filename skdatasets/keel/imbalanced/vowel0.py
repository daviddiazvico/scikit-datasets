"""
Keel vowel0 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_vowel0(return_X_y=False):
    """Load vowel0 dataset.

    Loads the vowel0 dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    return load_imbalanced('vowel0',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['TT', 'SpeakerNumber', 'Sex', 'F0', 'F1',
                                  'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                                  'F9', 'Class'], target_names=['Class'],
                           return_X_y=return_X_y)
