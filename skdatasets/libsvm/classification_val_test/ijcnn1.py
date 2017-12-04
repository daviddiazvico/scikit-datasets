"""
LIBSVM ijcnn1 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_train_val_test
from ...base import fetch_bz2


def load_ijcnn1(return_X_y=False):
    """Load ijcnn1 dataset.

    Loads the ijcnn1 dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test): lists of arrays
                                                            If return_X_y is
                                                            True

    """
    return load_train_val_test('ijcnn1',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2',
                               fetch_file=fetch_bz2, return_X_y=return_X_y)
