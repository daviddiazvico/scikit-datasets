"""
LIBSVM covtype.libsvm.binary dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_scale
from ...base import fetch_bz2


def load_covtype_binary(return_X_y=False):
    """Load covtype.binary dataset.

    Loads the covtype.binary dataset.

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
    return load_train_scale('covtype.binary',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2',
                            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2',
                            fetch_file=fetch_bz2, return_X_y=return_X_y)
