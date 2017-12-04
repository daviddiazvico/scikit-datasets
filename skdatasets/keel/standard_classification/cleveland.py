"""
Keel cleveland dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_standard_classification


def load_cleveland(return_X_y=False):
    """Load cleveland dataset.

    Loads the cleveland dataset.

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
    return load_standard_classification('cleveland',
                                        'http://sci2s.ugr.es/keel/dataset/data/classification',
                                        names=['Age', 'Sex', 'Cp', 'Trestbps',
                                               'Chol', 'Fbs', 'Restecg',
                                               'Thalach', 'Exang', 'Oldpeak',
                                               'Slope', 'Ca', 'Thal', 'Num'],
                                        target_names=['Num'],
                                        return_X_y=return_X_y)
