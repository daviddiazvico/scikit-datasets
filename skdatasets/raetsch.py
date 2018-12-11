"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""

import os
from scipy.io import loadmat
from sklearn.datasets.base import (_fetch_remote, get_data_home, Bunch,
                                   RemoteFileMetadata)


DATASETS = {'banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german',
            'heart', 'image', 'ringnorm', 'splice', 'thyroid', 'titanic',
            'twonorm', 'waveform'}
ARCHIVE = RemoteFileMetadata(filename='benchmarks.mat',
                             url='https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets/raw/master/benchmarks.mat',
                             checksum=('47c19e4bc4716edc4077cfa5ea61edf4d02af4ec51a0ecfe035626ae8b561c75'))


def fetch(name, data_home=None):
    """Fetch Gunnar Raetsch's dataset.

    Fetch a Gunnar Raetsch's benchmark dataset by name. Availabe datasets are
    'banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german', 'heart',
    'image', 'ringnorm', 'splice', 'thyroid', 'titanic', 'twonorm' and
    'waveform'. More info at
    https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets.

    Parameters
    ----------
    name : string
        Dataset name.
    data_home : string or None, default None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in ‘~/scikit_learn_data’ subfolders.

    Returns
    -------
    data : Bunch
        Dictionary-like object with all the data and metadata.

    """
    if name not in DATASETS:
        raise Exception('Avaliable datasets are ' + str(list(DATASETS)))
    dirname = os.path.join(get_data_home(data_home=data_home), 'raetsch')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = _fetch_remote(ARCHIVE, dirname=dirname)
    X, y, train_splits, test_splits = loadmat(filename)[name][0][0]
    cv = ((X[tr - 1], y[tr - 1], X[ts - 1], y[ts - 1]) for tr, ts in zip(train_splits, test_splits))
    return Bunch(data=X, target=y, data_test=None, target_test=None,
                 inner_cv=None, outer_cv=cv, DESCR=name)
