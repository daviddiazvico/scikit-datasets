"""
Test the LIBSVM loader.

@author: David Diaz Vico
@license: MIT
"""

from sklearn.model_selection import BaseCrossValidator

from skdatasets.libsvm import fetch_libsvm


def check(data, shape):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if hasattr(data, 'outer_cv') and (data.outer_cv is not None):
        assert len(list(data.outer_cv.split())) == 1
    if hasattr(data, 'inner_cv') and (data.inner_cv is not None):
        assert len(list(data.inner_cv.split())) == 1


def test_fetch_libsvm_australian():
    """Tests LIBSVM australian dataset."""
    data = fetch_libsvm('binary', 'australian')
    check(data, (690, 14))


def test_fetch_libsvm_liver_disorders():
    """Tests LIBSVM liver-disorders dataset."""
    data = fetch_libsvm('binary', 'liver-disorders')
    check(data, (145, 5))


def test_fetch_libsvm_duke():
    """Tests LIBSVM duke dataset."""
    data = fetch_libsvm('binary', 'duke')
    check(data, (44, 7129))


def test_fetch_libsvm_cod_rna():
    """Tests LIBSVM cod-rna dataset."""
    data = fetch_libsvm('binary', 'cod-rna')
    check(data, (59535, 8))


def test_fetch_libsvm_satimage():
    """Tests LIBSVM satimage dataset."""
    data = fetch_libsvm('multiclass', 'satimage.scale')
    check(data, (4435, 36))
