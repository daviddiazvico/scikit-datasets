"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Iterable, Tuple, Union

import numpy as np
from sacred.observers import FileStorageObserver
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch

from skdatasets.utils.experiment import experiment

if TYPE_CHECKING:
    IndicesType = np.typing.NDArray[int]
    ExplicitSplitType = Iterable[
        Tuple[
            np.typing.NDArray[float],
            np.typing.NDArray[Union[float, int]],
            np.typing.NDArray[float],
            np.typing.NDArray[Union[float, int]],
        ],
    ]
    if sys.version_info >= (3, 8):
        from typing import Protocol
    else:
        from typing_extensions import Protocol
else:
    IndicesType = np.ndarray
    Protocol = object


class CVSplitter(Protocol):

    def split(
        self,
        X: np.typing.NDArray[float],
        y: None = None,
        groups: None = None,
    ) -> Iterable[Tuple[IndicesType, IndicesType]]:
        pass

    def get_n_splits(
        self,
        X: np.typing.NDArray[float],
        y: None = None,
        groups: None = None,
    ) -> int:
        pass


CVLike = Union[
    CVSplitter,
    Iterable[Tuple[IndicesType, IndicesType]],
    int,
    None,
]


def _dataset(
    inner_cv: CVLike = None,
    outer_cv: CVLike = None,
) -> Bunch:
    data = load_boston()
    if outer_cv is None:
        X, X_test, y, y_test = train_test_split(data.data, data.target)
        data.data = X
        data.target = y
        data.data_test = X_test
        data.target_test = y_test
        data.outer_cv = None
    else:
        data.data_test = data.target_test = None
        data.outer_cv = outer_cv
    data.inner_cv = inner_cv

    return data


def _estimator(cv: CVLike) -> GridSearchCV:
    return GridSearchCV(
        DecisionTreeRegressor(),
        {'max_depth': [2, 4]},
        cv=cv,
    )


def _experiment(
    inner_cv: CVLike,
    outer_cv: CVLike | ExplicitSplitType,
) -> None:
    e = experiment(_dataset, _estimator)
    e.observers.append(FileStorageObserver('.results'))
    e.run(
        config_updates={
            'dataset': {
                'inner_cv': inner_cv,
                'outer_cv': outer_cv,
            },
        },
    )


def test_nested_cv() -> None:
    """Tests nested CV experiment."""
    _experiment(3, 3)


def test_inner_cv() -> None:
    """Tests inner CV experiment."""
    _experiment(3, None)


def test_explicit_inner_folds() -> None:
    """Tests explicit inner folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment(
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
        3,
    )


def test_explicit_outer_folds_indexes() -> None:
    """Tests explicit outer folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment(
        3,
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
    )


def test_explicit_outer_folds() -> None:
    """Tests explicit outer folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment(
        3,
        [
            (X[:10], y[:10], X[10:20], y[10:20]),
            (X[10:20], y[10:20], X[20:30], y[20:30]),
            (X[20:30], y[20:30], X[30:40], y[30:40]),
        ],
    )


def test_explicit_nested_folds() -> None:
    """Tests explicit nested folds experiment."""
    X, y = load_boston(return_X_y=True)
    _experiment(
        [
            (np.arange(3, 10), np.arange(3)),
            (np.concatenate((np.arange(3), np.arange(7, 10))), np.arange(3, 7)),
            (np.arange(7, 10), np.arange(7)),
        ],
        [
            (np.arange(10), np.arange(10, 20)),
            (np.arange(10, 20), np.arange(20, 30)),
            (np.arange(20, 30), np.arange(30, 40)),
        ],
    )
