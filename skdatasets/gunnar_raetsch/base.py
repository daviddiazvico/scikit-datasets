"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""

from scipy.io import loadmat
from sklearn.model_selection import BaseCrossValidator

from ..base import Bunch


class GunnarRaetschDatasetSplit(BaseCrossValidator):
    """Predefined split cross-validator for Gunnar Raetsch datasets.

    Provides train/test indices to split data into train/test sets using a
    predefined scheme.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    train_splits: array-like, shape (n_samples,)
                  List of indices for each training split.
    test_splits: array-like, shape (n_samples,)
                 List of indices for each test split.

    """

    def __init__(self, train_splits, test_splits):
        self.train_splits = train_splits - 1
        self.test_splits = test_splits - 1

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X: object
            Always ignored, exists for compatibility.
        y: object
            Always ignored, exists for compatibility.
        groups: object
            Always ignored, exists for compatibility.

        Returns
        -------
        train: ndarray
            The training set indices for that split.
        test: ndarray
            The testing set indices for that split.

        """
        for train_indices, test_indices in zip(self.train_splits, self.test_splits):
            yield (train_indices, test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X: object
           Always ignored, exists for compatibility.
        y: object
           Always ignored, exists for compatibility.
        groups: object
                Always ignored, exists for compatibility.

        Returns
        -------
        n_splits: int
                  Returns the number of splitting iterations in the
                  cross-validator.

        """
        return len(self.train_splits)


def load_dataset(name, return_X_y=False):
    """Load dataset.

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    features, target, train_splits, test_splits = loadmat('skdatasets/gunnar_raetsch/benchmarks')[name][0][0]

    if return_X_y:
        return features, target

    return Bunch(features=features, target=target,
                 splits=GunnarRaetschDatasetSplit(train_splits, test_splits))
