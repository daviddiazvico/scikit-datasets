"""
Scikit-learn datasets
(http://scikit-learn.org/stable/datasets/index.html).

@author: David Diaz Vico
@license: MIT
"""

from sklearn.datasets import (load_boston, load_breast_cancer, load_diabetes,
                              load_digits, load_iris, load_linnerud, load_wine)

datasets = {'boston': load_boston, 'breast_cancer': load_breast_cancer,
            'diabetes': load_diabetes, 'digits': load_digits,
            'iris': load_iris, 'linnerud': load_linnerud, 'wine': load_wine}


def load(name, return_X_y=False):
    """Load dataset.

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object.

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y, X_test, y_test, inner_cv, outer_cv: arrays
                                              If return_X_y is True

    """
    if return_X_y:
        X, y = datasets[name](return_X_y=True)
        return X, y, None, None, None, None
    return datasets[name](return_X_y=False)
