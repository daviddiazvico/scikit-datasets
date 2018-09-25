"""
Scikit-learn-compatible datasets.

@author: David Diaz Vico
@license: MIT
"""
from bz2 import decompress
from os import makedirs, remove
from os.path import basename, exists, join, normpath, splitext
from shutil import copyfileobj
import tarfile
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

from sklearn.datasets import get_data_home
from sklearn.model_selection import BaseCrossValidator


class CustomSplit(BaseCrossValidator):
    """Custom split cross-validator.

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
        self.train_splits = train_splits
        self.test_splits = test_splits

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
        for train_indices, test_indices in zip(self.train_splits,
                                               self.test_splits):
            yield (train_indices, test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

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


def fetch_file(dataname, urlname, data_home=None):
    """Fetch dataset.

    Fetch a file from a given url and stores it in a given directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    data_home: string, default=None
               Dataset directory.

    Returns
    -------
    filename: string
              Name of the file.

    """
    # check if this data set has been already downloaded
    data_home = get_data_home(data_home=data_home)
    data_home = join(data_home, dataname)
    if not exists(data_home):
        makedirs(data_home)
    filename = join(data_home, basename(normpath(urlname)))
    # if the file does not exist, download it
    if not exists(filename):
        try:
            data_url = urlopen(urlname)
        except HTTPError as e:
            if e.code == 404:
                e.msg = "Dataset '%s' not found." % dataname
            raise
        # store file
        try:
            with open(filename, 'w+b') as data_file:
                copyfileobj(data_url, data_file)
        except Exception:
            remove(filename)
            raise
        data_url.close()
    return filename


def fetch_zip(dataname, urlname, data_home=None):
    """Fetch zipped dataset.

    Fetch a zip file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    data_home: string, default=None
               Dataset directory.

    Returns
    -------
    data_home: string
               Directory.

    """
    # fetch file
    filename = fetch_file(dataname, urlname, data_home=data_home)
    data_home = get_data_home(data_home=data_home)
    data_home = join(data_home, dataname)
    # unzip file
    try:
        with ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(data_home)
    except Exception:
        remove(filename)
        raise
    return data_home


def fetch_tgz(dataname, urlname, data_home=None):
    """Fetch zipped dataset.

    Fetch a tgz file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    data_home: string, default=None
               Dataset directory.

    Returns
    -------
    data_home: string
               Directory.

    """
    # fetch file
    filename = fetch_file(dataname, urlname, data_home=data_home)
    data_home = get_data_home(data_home=data_home)
    data_home = join(data_home, dataname)
    # unzip file
    try:
        with tarfile.open(filename, 'r:gz') as tar_file:
            tar_file.extractall(data_home)
    except Exception:
        remove(filename)
        raise
    return data_home


def fetch_bz2(dataname, urlname, data_home=None):
    """Fetch bzipped dataset.

    Fetch a bz2 file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    data_home: string, default=None
               Dataset directory.

    Returns
    -------
    data_name: string
               Name of the file.

    """
    # fetch file
    filename = fetch_file(dataname, urlname, data_home=data_home)
    # unzip file
    try:
        with open(filename, 'rb') as compressed:
            decompressed = decompress(compressed.read())
        data_name = splitext(filename)[0]
        with open(data_name, 'w+b') as data_file:
            data_file.write(decompressed)
    except Exception:
        remove(filename)
        raise
    return data_name
