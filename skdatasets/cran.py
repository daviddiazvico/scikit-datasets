"""
Datasets extracted from R packages in CRAN (https://cran.r-project.org/).

@author: Carlos Ramos Carreño
@license: MIT
"""

from distutils.version import LooseVersion
from html.parser import HTMLParser
import os
from os import makedirs, remove
from os.path import basename, exists, join, normpath, splitext
import pathlib
import re
from shutil import copyfileobj
import tarfile
import urllib.request
from urllib.error import HTTPError
from urllib.request import urlopen
import warnings

from sklearn.datasets.base import Bunch, get_data_home, RemoteFileMetadata

import pandas as pd
import rdata


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


class _LatestVersionHTMLParser(HTMLParser):
    """
    Class for parsing the version in the CRAN package information page.
    """

    def __init__(self, *, convert_charrefs=True):
        HTMLParser.__init__(self, convert_charrefs=convert_charrefs)

        self.last_is_version = False
        self.version = None
        self.version_regex = re.compile('(?i).*version.*')
        self.handling_td = False

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.handling_td = True

    def handle_endtag(self, tag):
        self.handling_td = False

    def handle_data(self, data):
        if self.handling_td:
            if self.last_is_version:
                self.version = data
                self.last_is_version = False
            elif self.version_regex.match(data):
                self.last_is_version = True


def _get_latest_version_online(package_name):
    """
    Get the latest version of the package from CRAN.

    """
    parser = _LatestVersionHTMLParser()

    url_request = urllib.request.Request(
        url="https://CRAN.R-project.org/package=" + package_name)
    try:
        url_file = urllib.request.urlopen(url_request)
    except urllib.request.HTTPError as e:
        if e.code == 404:
            e.msg = f"Package '{package_name}' not found."
        raise
    url_content = url_file.read().decode('utf-8')

    parser.feed(url_content)

    return parser.version


def _get_latest_version_offline(package_name):
    """
    Get the latest downloaded version of the package.

    Returns None if not found.

    """
    home = pathlib.Path(get_data_home())  # Should allow providing data home?

    downloaded_packages = tuple(home.glob(package_name + "_*.tar.gz"))

    if downloaded_packages:
        versions = [
            LooseVersion(p.name[(len(package_name) + 1):-len(".tar.gz")])
            for p in downloaded_packages]

        versions.sort()
        latest_version = versions[-1]

        return str(latest_version)
    else:
        return None


def _get_version(package_name, *, version=None):
    """
    Get the version of the package.

    If the version is specified, return it.
    Otherwise, try to find the last version online.
    If offline, try to find the downloaded version, if any.

    """
    if version is None:
        try:
            version = _get_latest_version_online(package_name)
        except urllib.request.URLError:
            version = _get_latest_version_offline(package_name)

            if version is None:
                raise

    return version


def _get_urls(package_name, *, version=None):

    version = _get_version(package_name, version=version)

    latest_url = ("https://cran.r-project.org/src/contrib/" + package_name +
                  "_" + version + ".tar.gz")
    archive_url = ("https://cran.r-project.org/src/contrib/Archive/" +
                   package_name + "/" + package_name +
                   "_" + version + ".tar.gz")
    return (latest_url, archive_url)


def _download_package_data(package_name, *, package_url=None, version=None,
                           folder_name=None,
                           fetch_file=fetch_tgz, subdir=None):

    if package_url is None:
        url_list = _get_urls(package_name, version=version)
    else:
        url_list = (package_url,)

    if folder_name is None:
        folder_name = os.path.basename(url_list[0])

    if subdir is None:
        subdir = "data"

    for i, url in enumerate(url_list):
        try:
            directory = fetch_file(folder_name, url)
            break
        except Exception:
            # If it is the last url, reraise
            if i >= len(url_list) - 1:
                raise

    directory_path = pathlib.Path(directory)

    data_path = directory_path / package_name / subdir

    return data_path


def fetch_dataset(dataset_name, package_name, *, package_url=None,
                  version=None, folder_name=None, subdir=None,
                  fetch_file=fetch_tgz, converter=None):
    """Fetch an R dataset.

    Only .rda datasets in community packages can be downloaded for now.

    R datasets do not have a fixed structure, so this function does not
    attempt to force one.

    Parameters
    ----------
    dataset_name: string
        Name of the dataset, including extension if any.
    package_name: string
        Name of the R package where this dataset resides.
    package_url: string
        Package url. If `None` it tries to obtain it from the package name.
    version: string
        If `package_url` is not specified, the version of the package to
        download. By default is the latest one.
    folder_name: string
        Name of the folder where the downloaded package is stored. By default,
        is the last component of `package_url`.
    subdir: string
        Subdirectory of the package containing the datasets. By default is
        'data'.
    fetch_file: function, default=fetch_tgz
        Dataset fetching function.
    converter: rdata.conversion.Converter
        Object used to translate R objects into Python objects.

    Returns
    -------
    data: dict
        Dictionary-like object with all the data and metadata.

    """

    if converter is None:
        converter = rdata.conversion.SimpleConverter()

    data_path = _download_package_data(package_name, package_url=package_url,
                                       version=version,
                                       folder_name=folder_name,
                                       fetch_file=fetch_file,
                                       subdir=subdir)

    file_path = data_path / dataset_name

    if not file_path.suffix:
        possible_names = list(data_path.glob(dataset_name + ".*"))
        if len(possible_names) != 1:
            raise FileNotFoundError(f"Dataset {dataset_name} not found in "
                                    f"package {package_name}")
        dataset_name = possible_names[0]
        file_path = data_path / dataset_name

    parsed = rdata.parser.parse_file(file_path)

    converted = converter.convert(parsed)

    return converted


def fetch_package(package_name, *, package_url=None,
                  version=None,
                  folder_name=None, subdir=None,
                  fetch_file=fetch_tgz,
                  converter=None, ignore_errors=False):
    """Fetch all datasets from a R package.

    Only .rda datasets in community packages can be downloaded for now.

    R datasets do not have a fixed structure, so this function does not
    attempt to force one.

    Parameters
    ----------
    package_name: string
        Name of the R package.
    package_url: string
        Package url. If `None` it tries to obtain it from the package name.
    version: string
        If `package_url` is not specified, the version of the package to
        download. By default is the latest one.
    folder_name: string
        Name of the folder where the downloaded package is stored. By default,
        is the last component of `package_url`.
    subdir: string
        Subdirectory of the package containing the datasets. By default is
        'data'.
    fetch_file: function, default=fetch_tgz
        Dataset fetching function.
    converter: rdata.conversion.Converter
        Object used to translate R objects into Python objects.
    ignore_errors: boolean
        If True, ignore the datasets producing errors and return the
        remaining ones.

    Returns
    -------
    data: dict
        Dictionary-like object with all the data and metadata.

    """

    if converter is None:
        converter = rdata.conversion.SimpleConverter()

    data_path = _download_package_data(package_name, package_url=package_url,
                                       version=version,
                                       folder_name=folder_name,
                                       fetch_file=fetch_file,
                                       subdir=subdir)

    if not data_path.exists():
        return {}

    all_datasets = {}

    for dataset in data_path.iterdir():

        if dataset.suffix.lower() in ['.rda', '.rdata']:
            try:
                parsed = rdata.parser.parse_file(dataset)

                converted = converter.convert(parsed)

                all_datasets.update(converted)
            except Exception:
                if not ignore_errors:
                    raise
                else:
                    warnings.warn(f"Error loading dataset {dataset.name}",
                                  stacklevel=2)

    return all_datasets


datasets = {
    'geyser': {
        'load_args': (['geyser.rda', 'MASS'], {}),
        'sklearn_args': ([], {'target_name': 'waiting'})
        }
    }


def _to_sklearn(dataset, *, target_name):
    """Transforms R datasets to Sklearn format, if possible"""
    assert len(dataset.keys()) == 1
    name = tuple(dataset.keys())[0]
    obj = dataset[name]

    if isinstance(obj, pd.DataFrame):
        feature_names = list(obj.keys())
        feature_names.remove(target_name)
        X = pd.get_dummies(obj[feature_names]).values
        y = obj[target_name].values
    else:
        raise ValueError("Dataset not automatically convertible to "
                         "Sklearn format")

    return Bunch(data=X, target=y,
                 target_names=target_name, feature_names=feature_names)


def fetch_cran(name):
    """Load

    Load a dataset.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    data : Bunch
          Dictionary-like object with all the data and metadata.

    """
    load_args = datasets[name]['load_args']
    dataset = fetch_dataset(*load_args[0], **load_args[1])

    sklearn_args = datasets[name]['sklearn_args']
    sklearn_dataset = _to_sklearn(dataset, *sklearn_args[0], **sklearn_args[1])
    return sklearn_dataset
