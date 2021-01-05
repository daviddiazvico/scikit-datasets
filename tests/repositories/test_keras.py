"""
Test the Keras loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.keras import fetch

from . import check_estimator


def check(data, n_samples_train, n_samples_test, n_features):
    """Check dataset properties."""
    assert data.data.shape == (n_samples_train + n_samples_test, n_features)
    assert data.target.shape[0] == n_samples_train + n_samples_test
    assert data.train_indexes.shape == (n_samples_train,)
    assert data.test_indexes.shape == (n_samples_test,)
    assert not data.validation_indexes


def test_keras_mnist():
    """Tests keras MNIST dataset."""
    data = fetch('mnist')
    check(data, n_samples_train=60000, n_samples_test=10000, n_features=28 * 28)
