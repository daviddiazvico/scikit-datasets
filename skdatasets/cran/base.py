
from html.parser import HTMLParser
import os
import pathlib
import re
import urllib.request
import warnings

import rdata

from ..base import fetch_tgz


class LatestVersionHTMLParser(HTMLParser):

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


def _get_url(package_name):
    parser = LatestVersionHTMLParser()

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

    download_url = ("https://cran.r-project.org/src/contrib/" + package_name +
                    "_" + parser.version + ".tar.gz")
    return download_url


def _download_package_data(package_name, *, package_url=None, folder_name=None,
                           fetch_file=fetch_tgz, subdir=None):

    if package_url is None:
        package_url = _get_url(package_name)

    if folder_name is None:
        folder_name = os.path.basename(package_url)

    if subdir is None:
        subdir = "data"

    directory = fetch_file(folder_name, package_url)
    directory_path = pathlib.Path(directory)

    data_path = directory_path / package_name / subdir

    return data_path


def load_dataset(dataset_name, package_name, *, package_url=None,
                 folder_name=None, fetch_file=fetch_tgz,
                 converter=None, subdir=None):
    """Load an R dataset.

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
    folder_name: string
        Name of the folder where the downloaded package is stored. By default,
        is the last component of `package_url`.
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
                                       folder_name=folder_name,
                                       fetch_file=fetch_file,
                                       subdir=subdir)

    file_path = data_path / dataset_name

    parsed = rdata.parser.parse_file(file_path)

    converted = converter.convert(parsed)

    return converted


def load_package(package_name, *, package_url=None,
                 folder_name=None, fetch_file=fetch_tgz,
                 converter=None, ignore_errors=False,
                 subdir=None):
    """Load all datasets from a R package.

    Only .rda datasets in community packages can be downloaded for now.

    R datasets do not have a fixed structure, so this function does not
    attempt to force one.

    Parameters
    ----------
    package_name: string
        Name of the R package.
    package_url: string
        Package url. If `None` it tries to obtain it from the package name.
    folder_name: string
        Name of the folder where the downloaded package is stored. By default,
        is the last component of `package_url`.
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
