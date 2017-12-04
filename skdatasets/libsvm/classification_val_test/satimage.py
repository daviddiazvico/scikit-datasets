"""
LIBSVM satimage dataset.

@author: David Diaz Vico
@license: MIT
"""


from ..base import load_train_val_test


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
    (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test): lists of arrays
                                                            If return_X_y is
                                                            True

    """
    return load_train_val_test('satimage',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.val',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t',
                               return_X_y=return_X_y)
