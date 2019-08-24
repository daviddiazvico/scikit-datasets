"""
Test the Keras loader.

@author: David Diaz Vico
@license: MIT
"""

from . import check_estimator

from skdatasets.repositories.keras import fetch


def check(data, shape, test_shape):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    assert data.data_test.shape == test_shape 
    assert data.target_test.shape[0] == test_shape[0] 


def test_keras_mnist():
    """Tests keras MNIST dataset."""
    data = fetch('mnist')
    check(data, (60000, 28*28), (10000, 28*28))
