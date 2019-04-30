"""
Datasets from the UCR time series database.

@author: Carlos Ramos Carreño
@license: MIT
"""

from os.path import basename, normpath
import pathlib
from shutil import copyfileobj
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile

import scipy.io.arff
from sklearn.datasets.base import Bunch, get_data_home

import numpy as np

BASE_URL = 'http://www.timeseriesclassification.com/Downloads/'


def fetch_file(dataname, urlname, subfolder=None, data_home=None):
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
    data_home = pathlib.Path(get_data_home(data_home=data_home))

    if subfolder:
        data_home = data_home / subfolder

    data_home = data_home / dataname
    if not data_home.exists():
        data_home.mkdir(parents=True)
    filename = data_home / basename(normpath(urlname))
    # if the file does not exist, download it
    if not filename.exists():
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
            filename.unlink()
            raise
        data_url.close()
    return filename


def fetch_zip(dataname, urlname, subfolder=None, data_home=None):
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
    filename = fetch_file(dataname, urlname, subfolder=subfolder,
                          data_home=data_home)
    data_home = filename.parent
    # unzip file
    try:
        with zipfile.ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(data_home)
    except Exception:
        filename.unlink()
        raise
    return data_home


def _target_conversion(target):
    try:
        target_data = target.astype(int)
        target_names = np.unique(target_data).astype(str).tolist()
    except ValueError:
        target_names = np.unique(target).tolist()
        target_data = np.searchsorted(target_names, target)

    return target_data, target_names


def data_to_matrix(struct_array):
    if(len(struct_array.dtype.fields.items()) == 1 and
       list(struct_array.dtype.fields.items())[0][1][0] == np.object_):
        attribute = struct_array[list(
            struct_array.dtype.fields.items())[0][0]]

        n_instances = len(attribute)
        n_curves = len(attribute[0])
        n_points = len(attribute[0][0])

        attribute_new = np.zeros(n_instances, dtype=np.object_)

        for i in range(n_instances):

            transformed_matrix = np.zeros((n_curves, n_points))

            for j in range(n_curves):
                for k in range(n_points):
                    transformed_matrix[j][k] = attribute[i][j][k]
                    attribute_new[i] = transformed_matrix

        return attribute_new

    else:
        return np.array(struct_array.tolist())


def fetch(name, data_home=None):
    url = BASE_URL + name

    data_home = fetch_zip(name, urlname=url + '.zip', subfolder="ucr",
                          data_home=data_home)

    description_filenames = [name, name + "Description", name + "_Info"]

    for f in description_filenames:
        path_file_descr = (data_home / f).with_suffix(".txt")
        if path_file_descr.exists():
            break
    else:
        # No description is found
        path_file_descr = None

    path_file_train = (data_home / (name + '_TRAIN')).with_suffix(".arff")
    path_file_test = (data_home / (name + '_TEST')).with_suffix(".arff")

    DESCR = (path_file_descr.read_text(errors='surrogateescape')
             if path_file_descr else '')
    train = scipy.io.arff.loadarff(path_file_train)
    test = scipy.io.arff.loadarff(path_file_test)
    dataset_name = train[1].name
    column_names = np.array(train[1].names())
    target_column_name = column_names[-1]
    feature_names = column_names[column_names != target_column_name].tolist()
    target_column = train[0][target_column_name].astype(str)
    test_target_column = test[0][target_column_name].astype(str)
    target, target_names = _target_conversion(target_column)
    target_test, target_names_test = _target_conversion(test_target_column)
    assert target_names == target_names_test
    data = data_to_matrix(train[0][feature_names])
    data_test = data_to_matrix(test[0][feature_names])

    return Bunch(data=data, target=target,
                 data_test=data_test, target_test=target_test,
                 name=dataset_name, DESCR=DESCR,
                 feature_names=feature_names,
                 target_names=target_names)
