"""
Keel led7digit-0-2-4-5-6-7-8-9_vs_1 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_led7digit_0_2_4_5_6_7_8_9_vs_1(return_X_y=False):
    """Load led7digit-0-2-4-5-6-7-8-9_vs_1 dataset.

    Loads the led7digit-0-2-4-5-6-7-8-9_vs_1 dataset.

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
    return load_imbalanced('led7digit-0-2-4-5-6-7-8-9_vs_1',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                           names=['Led1', 'Led2', 'Led3', 'Led4', 'Led5',
                                  'Led6', 'Led7', 'number'],
                           target_names=['number'], return_X_y=return_X_y)
