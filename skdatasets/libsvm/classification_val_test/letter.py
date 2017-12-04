"""
LIBSVM letter dataset.

@author: David Diaz Vico
@license: MIT
"""


from ..base import load_train_val_test


def load_letter(return_X_y=False):
    """Load letter dataset.

    Loads the letter dataset.

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
    return load_train_val_test('letter',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.tr',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.val',
                               'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t',
                               return_X_y=return_X_y)
