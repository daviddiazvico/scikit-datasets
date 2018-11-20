"""
Test the Keras loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.keras import fetch_keras


def check(data, shape):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    assert len(list(data.outer_cv.split())) == 1


def test_keras_mnist():
    """Tests keras MNIST dataset."""
    data = fetch_keras('mnist')
    check(data, (60000, 28*28))
