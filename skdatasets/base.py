"""
Common utilities.
"""

from os.path import basename, normpath
import pathlib
from shutil import copyfileobj
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile
import tarfile
from sklearn.datasets.base import get_data_home


def fetch_file(dataname, urlname, subfolder=None, data_home=None):
    """Fetch dataset.

    Fetch a file from a given url and stores it in a given directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    subfolder: string, default=None
               The subfolder where to put the data, if any.
    data_home: string, default=None
               Dataset directory. If None, use the default of scikit-learn.

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


def fetch_compressed(dataname, urlname, compression_open,
                     subfolder=None, data_home=None, open_format='r'):
    """Fetch compressed dataset.

    Fetch a compressed file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    compression_open: callable
                      Module/class used to decompress the data.
    subfolder: string, default=None
               The subfolder where to put the data, if any.
    data_home: string, default=None
               Dataset directory. If None, use the default of scikit-learn.
    open_format: string
                 Format for opening the compressed file.

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
        with compression_open(filename, open_format) as compressed_file:
            compressed_file.extractall(data_home)
    except Exception:
        filename.unlink()
        raise
    return data_home


def fetch_zip(dataname, urlname, subfolder=None, data_home=None):
    """Fetch zipped dataset.

    Fetch a zip file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    subfolder: string, default=None
               The subfolder where to put the data, if any.
    data_home: string, default=None
               Dataset directory. If None, use the default of scikit-learn.

    Returns
    -------
    data_home: string
               Directory.

    """
    return fetch_compressed(dataname=dataname, urlname=urlname,
                            compression_open=zipfile.ZipFile,
                            subfolder=subfolder,
                            data_home=data_home)


def fetch_tgz(dataname, urlname, subfolder=None, data_home=None):
    """Fetch tgz dataset.

    Fetch a tgz file from a given url, unzips and stores it in a given
    directory.

    Parameters
    ----------
    dataname: string
              Dataset name.
    urlname: string
             Dataset url.
    subfolder: string, default=None
               The subfolder where to put the data, if any.
    data_home: string, default=None
               Dataset directory. If None, use the default of scikit-learn.

    Returns
    -------
    data_home: string
               Directory.

    """
    return fetch_compressed(dataname=dataname, urlname=urlname,
                            compression_open=tarfile.open,
                            subfolder=subfolder,
                            data_home=data_home,
                            open_format='r:gz')
