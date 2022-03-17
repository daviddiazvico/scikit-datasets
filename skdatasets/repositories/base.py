"""
Common utilities.
"""

import pathlib
import tarfile
import zipfile
from os.path import basename, normpath
from shutil import copyfileobj
from typing import Callable, Optional, Sequence, Union, overload
from urllib.error import HTTPError
from urllib.request import urlopen

from sklearn.datasets import get_data_home

CompressedFile = Union[zipfile.ZipFile, tarfile.TarFile]

OpenMethod = Callable[
    [pathlib.Path, str],
    CompressedFile,
]


def fetch_file(
    dataname: str,
    urlname: str,
    subfolder: Optional[str] = None,
    data_home: Optional[str] = None,
) -> pathlib.Path:
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
    filename: Path
              Name of the file.

    """
    # check if this data set has been already downloaded
    data_home_path = pathlib.Path(get_data_home(data_home=data_home))

    if subfolder:
        data_home_path /= subfolder

    data_home_path /= dataname
    if not data_home_path.exists():
        data_home_path.mkdir(parents=True)
    filename = data_home_path / basename(normpath(urlname))
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


@overload
def _missing_files(
    compressed_file: zipfile.ZipFile,
    data_home_path: pathlib.Path,
) -> Sequence[zipfile.ZipInfo]:
    pass


@overload
def _missing_files(
    compressed_file: tarfile.TarFile,
    data_home_path: pathlib.Path,
) -> Sequence[tarfile.TarInfo]:
    pass


def _missing_files(
    compressed_file: CompressedFile,
    data_home_path: pathlib.Path,
) -> Sequence[Union[zipfile.ZipInfo, tarfile.TarInfo]]:

    if isinstance(compressed_file, zipfile.ZipFile):

        members_zip = compressed_file.infolist()

        return [
            info for info in members_zip
            if not (data_home_path / info.filename).exists()
        ]

    members_tar = compressed_file.getmembers()

    return [
        info for info in members_tar
        if not (data_home_path / info.name).exists()
    ]


def fetch_compressed(
    dataname: str,
    urlname: str,
    compression_open: OpenMethod,
    subfolder: Optional[str] = None,
    data_home: Optional[str] = None,
    open_format: str = 'r',
) -> pathlib.Path:
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
    data_home: Path
               Directory.

    """
    # fetch file
    filename = fetch_file(
        dataname,
        urlname,
        subfolder=subfolder,
        data_home=data_home,
    )
    data_home_path = filename.parent
    # unzip file
    try:
        with compression_open(filename, open_format) as compressed_file:
            compressed_file.extractall(
                data_home_path,
                members=_missing_files(compressed_file, data_home_path),
            )
    except Exception:
        filename.unlink()
        raise
    return data_home_path


def fetch_zip(
    dataname: str,
    urlname: str,
    subfolder: Optional[str] = None,
    data_home: Optional[str] = None,
) -> pathlib.Path:
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
    data_home: Path
               Directory.

    """
    return fetch_compressed(
        dataname=dataname,
        urlname=urlname,
        compression_open=zipfile.ZipFile,
        subfolder=subfolder,
        data_home=data_home,
    )


def fetch_tgz(
    dataname: str,
    urlname: str,
    subfolder: Optional[str] = None,
    data_home: Optional[str] = None,
) -> pathlib.Path:
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
    data_home: Path
               Directory.

    """
    return fetch_compressed(
        dataname=dataname,
        urlname=urlname,
        compression_open=tarfile.open,
        subfolder=subfolder,
        data_home=data_home,
        open_format='r:gz',
    )
