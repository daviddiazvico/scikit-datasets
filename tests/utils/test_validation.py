"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.datasets import load_boston, load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

from skdatasets.utils.validation import scatter_plot, metaparameter_plot


def test_scatter_plot():
    """Tests scatter plot."""
    X, y = load_boston(return_X_y=True)
    estimator = PCA(n_components=10)
    estimator.fit(X, y)
    image_files = scatter_plot(X, y, estimator)
    assert len(image_files) == 1
    estimator = DummyRegressor()
    estimator.fit(X, y)
    image_files = scatter_plot(X, y, estimator)
    assert len(image_files) == 1
    X, y = load_iris(return_X_y=True)
    estimator = LinearDiscriminantAnalysis()
    estimator.fit(X, y)
    image_files = scatter_plot(X, y, estimator)
    assert len(image_files) == 2


def test_metaparameter_plot():
    """Tests metaparameter plot."""
    X, y = load_boston(return_X_y=True)
    estimator = GridSearchCV(DummyRegressor(),
                             {'strategy': ['mean', 'median', 'constant'],
                              'constant': [1.0, 2.0, 3.0]})
    estimator.fit(X, y)
    image_files = metaparameter_plot(estimator)
    assert len(image_files) == 1
