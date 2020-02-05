"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""

import hashlib
import os
from scipy.io import loadmat
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch
from urllib.request import urlretrieve


DATASETS = {'banana', 'breast_cancer', 'diabetis', 'flare_solar', 'german',
            'heart', 'image', 'ringnorm', 'splice', 'thyroid', 'titanic',
            'twonorm', 'waveform'}


def _fetch_remote(dirname=None):
    """Helper function to download the remote dataset into path

    Fetch the remote dataset, save into path using remote's filename and ensure
    its integrity based on the SHA256 Checksum of the downloaded file.

    Parameters
    ----------
    dirname : string
        Directory to save the file to.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """
    file_path = 'benchmarks.mat'
    if dirname is not None:
        file_path = os.path.join(dirname, file_path)
    urlretrieve('https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets/raw/master/benchmarks.mat', file_path)
    sha256hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(8192)
            if not buffer:
                break
            sha256hash.update(buffer)
    checksum = sha256hash.hexdigest()
    remote_checksum = ('47c19e4bc4716edc4077cfa5ea61edf4d02af4ec51a0ecfe035626ae8b561c75')
    if remote_checksum != checksum:
        raise IOError(f"{file_path} has an SHA256 checksum ({checksum}) differing from expected ({remote_checksum}), file may be corrupted.")
    return file_path


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
    filename = _fetch_remote(dirname=dirname)
    X, y, train_splits, test_splits = loadmat(filename)[name][0][0]
    cv = ((X[tr - 1], y[tr - 1], X[ts - 1], y[ts - 1]) for tr, ts in zip(train_splits, test_splits))
    return Bunch(data=X, target=y, data_test=None, target_test=None,
                 inner_cv=None, outer_cv=cv, DESCR=name)
