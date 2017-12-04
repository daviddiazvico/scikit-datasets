"""
Forex/Bitcoin datasets (http://forex-python.readthedocs.io/en/latest/).

@author: David Diaz Vico
@license: MIT
"""

from datetime import date, timedelta

from ..base import Bunch


def load(get_rate, fdescr='', start=date(2015, 1, 1), end=date(2015, 1, 31),
         return_X_y=False):
    """Load dataset.

    Loads the dataset.

    Parameters
    ----------
    get_rate: function
              Dataset loading function.
    fdescr: string, default=''
            Dataset description.
    start: date, default=2015-1-1
           Starting date.
    end: date, default=2015-1-31
         Ending date.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object.

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X: array
       If return_X_y is True

    """
    features = []
    days = []
    delta = end - start
    for d in range(delta.days + 1):
        day = start + timedelta(days=d)
        rate = get_rate(day)
        features.append(rate)
        days.append(day.strftime('%Y-%m-%d'))

    if return_X_y:
        return features

    return Bunch(data=features, DESCR=fdescr, feature_names=days)
