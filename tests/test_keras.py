"""
Test the Keras loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.keras import fetch_keras


def check(data, shape, splits=1):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if splits > 1:
        assert len(list(data.outer_cv.split())) == splits


def test_keras_mnist():
    """Tests keras MNIST dataset."""
    data = fetch_keras('mnist')
    check(data, (60000, 28*28), 1)
