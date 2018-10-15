"""
Test the LIBSVM loader.

@author: David Diaz Vico
@license: MIT
"""

from sklearn.model_selection import BaseCrossValidator

from skdatasets.libsvm import fetch_libsvm


def check(data, shape, test_shape=None):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if test_shape is not None:
        assert data.data_test.shape == test_shape
        assert data.target_test.shape[0] == test_shape[0]
    if hasattr(data, 'inner_cv'):
        assert isinstance(data.inner_cv, BaseCrossValidator)


def test_fetch_libsvm_australian():
    """Tests LIBSVM australian dataset."""
    data = fetch_libsvm(collection='binary', name='australian')
    check(data, (1380, 14))


def test_fetch_libsvm_liver_disorders():
    """Tests LIBSVM liver-disorders dataset."""
    data = fetch_libsvm(collection='binary', name='liver-disorders')
    check(data, (290, 5), test_shape=(145, 5))


def test_fetch_libsvm_duke():
    """Tests LIBSVM duke dataset."""
    data = fetch_libsvm(collection='binary', name='duke')
    check(data, (88, 7129))


def test_fetch_libsvm_cod_rna():
    """Tests LIBSVM cod-rna dataset."""
    data = fetch_libsvm(collection='binary', name='cod-rna')
    check(data, (119070, 8))


def test_fetch_libsvm_satimage():
    """Tests LIBSVM satimage dataset."""
    data = fetch_libsvm(collection='multiclass', name='satimage.scale')
    check(data, (8870, 36), test_shape=(4435, 36))
