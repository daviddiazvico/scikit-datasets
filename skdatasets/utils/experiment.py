"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools
import os
import sys
from contextlib import contextmanager
from inspect import signature
from tempfile import NamedTemporaryFile, mkdtemp
from time import perf_counter, process_time
from typing import (
    IO,
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

import numpy as np
from sacred import Experiment, Ingredient
from sacred.observers import RunObserver
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils import Bunch, is_scalar_nan

SelfType = TypeVar("SelfType")


class DataLike(Protocol):

    def __getitem__(
        self: SelfType,
        key: np.typing.NDArray[int],
    ) -> SelfType:
        pass

    def __len__(self) -> int:
        pass


DataType = TypeVar("DataType", bound=DataLike, contravariant=True)
TargetType = TypeVar("TargetType", bound=DataLike)
IndicesType = Tuple[np.typing.NDArray[int], np.typing.NDArray[int]]
ExplicitSplitType = Tuple[
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int]],
    np.typing.NDArray[float],
    np.typing.NDArray[Union[float, int]],
]


class EstimatorProtocol(Protocol[DataType, TargetType]):

    def fit(self: SelfType, X: DataType, y: TargetType) -> SelfType:
        pass

    def predict(self, X: DataType) -> TargetType:
        pass


class CVSplitter(Protocol):

    def split(
        self,
        X: np.typing.NDArray[float],
        y: None = None,
        groups: None = None,
    ) -> Iterable[IndicesType]:
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
    Iterable[IndicesType],
    int,
    None,
]


def _append_info(experiment: Experiment, name: str, value: Any) -> None:
    info_list = experiment.info.get(name, [])
    info_list.append(value)
    experiment.info[name] = info_list


@contextmanager
def _add_timing(experiment: Experiment, name: str) -> Iterator[None]:
    initial_time = perf_counter()
    try:
        yield None
    finally:
        final_time = perf_counter()
        elapsed_time = final_time - initial_time
        _append_info(experiment, name, elapsed_time)


def _iterate_outer_cv(
    outer_cv: CVLike | Iterable[
        Tuple[DataType, TargetType, DataType, TargetType]
    ],
    estimator: EstimatorProtocol[DataType, TargetType],
    X: DataType,
    y: TargetType,
) -> Iterable[Tuple[DataType, TargetType, DataType, TargetType]]:
    """Iterate over multiple partitions."""
    if isinstance(outer_cv, Iterable):
        outer_cv, cv_copy = itertools.tee(outer_cv)
        if len(next(cv_copy)) == 4:
            yield from outer_cv

    cv = check_cv(outer_cv, y, classifier=is_classifier(estimator))
    yield from (
        (X[train], y[train], X[test], y[test])
        for train, test in cv.split(X, y)
    )


def _benchmark_from_data(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    X_train: DataType,
    y_train: TargetType,
    X_test: DataType,
    y_test: TargetType,
    save_train: bool = False,
) -> None:
    with _add_timing(experiment, "fit_time"):
        estimator.fit(X_train, y_train)

    _append_info(experiment, "fitted_estimator", estimator)

    with _add_timing(experiment, "score_time"):
        test_score = estimator.score(X_test, y_test)

    _append_info(experiment, "test_score", test_score)

    if save_train:
        train_score = estimator.score(X_train, y_train)
        _append_info(experiment, "train_score", train_score)

    for output in ("transform", "predict"):
        method = getattr(estimator, output, None)
        if method is not None:
            with _add_timing(experiment, f"{output}_time"):
                _append_info(experiment, f"{output}", method(X_test))


def _benchmark_one(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    data: Bunch,
    save_train: bool = False,
) -> None:
    """Use only one predefined partition."""
    X = data.data
    y = data.target

    train_indices = getattr(data, "train_indices", [])
    validation_indices = getattr(data, "validation_indices", [])
    test_indices = getattr(data, "test_indices", [])

    X_train_val = (
        X[train_indices + validation_indices]
        if train_indices
        else X
    )
    y_train_val = (
        y[train_indices + validation_indices]
        if train_indices
        else y
    )

    X_test = X[test_indices]
    y_test = y[test_indices]

    _benchmark_from_data(
        experiment=experiment,
        estimator=estimator,
        X_train=X_train_val,
        y_train=y_train_val,
        X_test=X_test,
        y_test=y_test,
        save_train=save_train,
    )

    experiment.info["score_mean"] = np.nanmean(experiment.info["test_score"])
    experiment.info["score_std"] = np.nanstd(experiment.info["test_score"])


def _benchmark_partitions(
    experiment: Experiment,
    *,
    estimator: BaseEstimator,
    data: Bunch,
    save_train: bool = False,
) -> None:
    """Use several partitions."""
    for X_train, y_train, X_test, y_test in _iterate_outer_cv(
        outer_cv=data.outer_cv,
        estimator=estimator,
        X=data.data,
        y=data.target,
    ):

        _benchmark_from_data(
            experiment=experiment,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            save_train=save_train,
        )

    experiment.info["score_mean"] = np.nanmean(experiment.info["test_score"])
    experiment.info["score_std"] = np.nanstd(experiment.info["test_score"])


def experiment(
    dataset: Callable[..., Bunch],
    estimator: Callable[..., BaseEstimator],
    *,
    save_train: bool = False,
) -> Experiment:
    """
    Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    dataset : function
        Dataset fetch function. Might receive any argument. Must return a
        :external:obj:`Bunch` with ``data``, ``target`` (might be ``None``),
        ``inner_cv`` (might be ``None``) and ``outer_cv``
        (might be ``None``).
    estimator : function
        Estimator initialization function. Might receive any keyword argument.
        Must return an initialized sklearn-compatible estimator.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """
    dataset_ingredient = Ingredient("dataset")
    dataset = dataset_ingredient.capture(dataset)
    estimator_ingredient = Ingredient("estimator")
    estimator = estimator_ingredient.capture(estimator)
    experiment = Experiment(
        ingredients=(
            dataset_ingredient,
            estimator_ingredient,
        ),
    )

    @experiment.automain
    def run() -> None:
        """Run the experiment."""
        data = dataset()

        # Metaparameter search
        cv = getattr(data, "inner_cv", None)

        try:
            e = estimator(cv=cv)
        except TypeError as exception:
            warn(f"The estimator does not accept cv: {exception}")
            e = estimator()

        # Model assessment
        if getattr(data, "test_indices", None):
            _benchmark_one(
                experiment=experiment,
                estimator=e,
                data=data,
                save_train=save_train,
            )
        elif getattr(data, "outer_cv", None) is not None:
            _benchmark_partitions(
                experiment=experiment,
                estimator=e,
                data=data,
                save_train=save_train,
            )

    return experiment
