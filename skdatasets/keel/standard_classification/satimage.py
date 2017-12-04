"""
Keel satimage dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_satimage(return_X_y=False):
    """Load satimage dataset.

    Loads the satimage dataset.

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
    return load_standard_classification('satimage',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['Sp11', 'Sp12', 'Sp13', 'Sp14',
                                               'Sp15', 'Sp16', 'Sp17', 'Sp18',
                                               'Sp19', 'Sp21', 'Sp22', 'Sp23',
                                               'Sp24', 'Sp25', 'Sp26', 'Sp27',
                                               'Sp28', 'Sp29', 'Sp31', 'Sp32',
                                               'Sp33', 'Sp34', 'Sp35', 'Sp36',
                                               'Sp37', 'Sp38', 'Sp39', 'Sp41',
                                               'Sp42', 'Sp43', 'Sp44', 'Sp45',
                                               'Sp46', 'Sp47', 'Sp48', 'Sp49',
                                               'Class'], target_names=['Class'],
                                        return_X_y=return_X_y)
