"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

@author: David Diaz Vico
@license: MIT
"""

from ..base import Bunch, fetch_file, load_svmlight_file, load_svmlight_files


def load_train(name, url, fetch_file=fetch_file, return_X_y=False):
    """Load dataset.

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    filename = fetch_file(name, url)
    X, y = load_svmlight_file(filename)
    X = X.todense()

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y)


def load_train_scale(name, url, url_scale, fetch_file=fetch_file,
                     return_X_y=False):
    """Load dataset with scaled version.

    Load a dataset with scaled version.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    url_scale: string
               Scaled dataset url.
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    filename = fetch_file(name, url)
    filename_scale = fetch_file(name, url_scale)
    X, y, X_scale, y_scale = load_svmlight_files([filename, filename_scale])
    X = X.todense()
    X_scale = X_scale.todense()

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, data_scale=X_scale, target_scale=y_scale)


def load_train_test(name, url, url_test, fetch_file=fetch_file,
                    return_X_y=False):
    """Load dataset with test partition.

    Load a dataset with test partition.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    url_test: string
              Test dataset url.
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    X, y, X_test, y_test = load_svmlight_files([filename, filename_test])
    X = X.todense()
    X_test = X_test.todense()

    if return_X_y:
        return (X, y), (X_test, y_test)

    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test)


def load_train_test_remaining(name, url, url_test, url_remaining,
                              fetch_file=fetch_file, return_X_y=False):
    """Load dataset with test and remaining partitions.

    Load a dataset with test and remaining partitions.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    url_test: string
              Test dataset url.
    url_remaining: string
                   Remaining dataset url.
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    filename_remaining = fetch_file(name, url_remaining)
    X, y, X_test, y_test, X_remaining, y_remaining = load_svmlight_files([filename,
                                                                          filename_test,
                                                                          filename_remaining])
    X = X.todense()
    X_test = X_test.todense()
    X_remaining = X_remaining.todense()

    if return_X_y:
        return (X, y), (X_test, y_test)

    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 data_remaining=X_remaining, target_remaining=y_remaining)


def load_train_val_test(name, url, url_tr, url_val, url_test,
                        fetch_file=fetch_file, return_X_y=False):
    """Load dataset with train, validation and test partitions.

    Load a dataset with train, validation and test partitions.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    url_tr: string
            Train dataset url.
    url_val: string
             Validation dataset url.
    url_test: string
              Test dataset url.
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test): lists of arrays
                                                            If return_X_y is
                                                            True

    """
    filename = fetch_file(name, url)
    filename_tr = fetch_file(name, url_tr)
    filename_val = fetch_file(name, url_val)
    filename_test = fetch_file(name, url_test)
    X, y, X_tr, y_tr, X_val, y_val, X_test, y_test = load_svmlight_files([filename,
                                                                          filename_tr,
                                                                          filename_val,
                                                                          filename_test])
    X = X.todense()
    X_tr = X_tr.todense()
    X_val = X_val.todense()
    X_test = X_test.todense()

    if return_X_y:
        return (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test)

    return Bunch(data=X, target=y, data_tr=X_tr, target_tr=y_tr, data_val=X_val,
                 target_val=y_val, data_test=X_test, target_test=y_test)


def load_train_test_scale(name, url, url_test, url_scale, url_test_scale,
                          fetch_file=fetch_file, return_X_y=False):
    """Load dataset with test partition and scaled version.

    Load a dataset with test partition and scaled version.

    Parameters
    ----------
    name: string
          Dataset name.
    url: string
         Dataset url.
    url_test: string
              Test dataset url.
    url_scale: string
               Scaled dataset url.
    url_test_scale: string
                    Scaled test dataset url
    fetch_file: function, default=fetch_file
                Dataset fetching function.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    (X, y), (X_test, y_test): lists of arrays
                              If return_X_y is True

    """
    filename = fetch_file(name, url)
    filename_test = fetch_file(name, url_test)
    filename_scale = fetch_file(name, url_scale)
    filename_scale_test = fetch_file(name, url_test_scale)
    X, y, X_test, y_test, X_scale, y_scale, X_scale_test, y_scale_test = load_svmlight_files([filename,
                                                                                              filename_test,
                                                                                              filename_scale,
                                                                                              filename_scale_test])
    X = X.todense()
    X_test = X_test.todense()
    X_scale = X_scale.todense()
    X_scale_test = X_scale_test.todense()

    if return_X_y:
        return (X, y), (X_test, y_test)

    return Bunch(data=X, target=y, data_test=X_test, target_test=y_test,
                 data_scale=X_scale, target_scale=y_scale,
                 data_scale_test=X_scale_test, target_scale_test=y_scale_test)
