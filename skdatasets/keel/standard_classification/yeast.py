"""
Keel yeast dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_yeast(return_X_y=False):
    """Load yeast dataset.

    Loads the yeast dataset.

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
    return load_standard_classification('yeast',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['Mcg', 'Gvh', 'Alm', 'Mit',
                                               'Erl', 'Pox', 'Vac', 'Nuc',
                                               'Class'], target_names=['Class'],
                                        return_X_y=return_X_y)
