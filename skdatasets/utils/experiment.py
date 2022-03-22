"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools
import os
from tempfile import NamedTemporaryFile, mkdtemp
from time import process_time
from typing import IO, Callable, Iterable, List, Mapping
from warnings import warn

import joblib
import numpy as np
from sacred import Experiment, Ingredient
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.utils import Bunch


def _benchmark_one(
    experiment: Experiment,
    data: Bunch,
    estimator: BaseEstimator,
) -> None:

    X = data.data
    y = data.target

    train_indices = getattr(data, 'train_indices', [])
    validation_indices = getattr(data, 'validation_indices', [])
    test_indices = getattr(data, 'test_indices', [])

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

    experiment.fit(X_train_val, y_train_val)
    try:
        with NamedTemporaryFile() as tmpfile:
            joblib.dump(experiment, tmpfile.name)
            experiment.add_artifact(
                tmpfile.name,
                name='estimator.joblib',
            )
    except Exception as exception:
        warn(f'Artifact save failed: {exception}')
    experiment.log_scalar(
        'score_mean',
        experiment.score(X_test, y_test),
    )
    experiment.log_scalar('score_std', 0.0)
    for output in ('transform', 'predict'):
        if hasattr(experiment, output):
            with open(
                os.path.join(
                    mkdtemp(),
                    f'{output}.npy',
                ),
                'wb+',
            ) as tmpfile:
                np.save(tmpfile, getattr(experiment, output)(X_test))
                experiment.add_artifact(tmpfile.name)


def _benchmark_partitions(
    experiment: Experiment,
    data: Bunch,
    estimator: BaseEstimator,
) -> None:
    # Outer CV
    # Explicit CV folds
    if isinstance(data.outer_cv, Iterable):
        cv, cv_copy = itertools.tee(data.outer_cv)
        if len(next(cv_copy)) == 4:
            scores: Mapping[str, List[float]] = {
                'test_score': [],
                'train_score': [],
                'fit_time': [],
                'score_time': [],
                'estimator': [],
            }
            outputs: Mapping[str, List[np.typing.NDArray[float]]] = {
                'transform': [],
                'predict': [],
            }
            for X, y, X_test, y_test in cv:
                t0 = process_time()
                estimator.fit(X, y=y)
                t1 = process_time()
                test_score = estimator.score(X_test, y=y_test)
                t2 = process_time()
                scores['test_score'].append(test_score)
                scores['train_score'].append(estimator.score(X, y=y))
                scores['fit_time'].append(t1 - t0)
                scores['score_time'].append(t2 - t1)
                scores['estimator'].append(estimator)
                for output in ('transform', 'predict'):
                    if hasattr(estimator, output):
                        outputs[output].append(
                            getattr(estimator, output)(X_test),
                        )

            tmpfile: IO[bytes]
            for output in ('transform', 'predict'):
                if outputs[output]:
                    with open(
                        os.path.join(
                            mkdtemp(),
                            f'{output}.npy',
                        ),
                        'wb+',
                    ) as tmpfile:
                        np.save(tmpfile, np.array(outputs[output]))
                        experiment.add_artifact(tmpfile.name)
            return
    else:
        cv = data.outer_cv

    # Automatic/indexed CV folds
    scores = cross_validate(
        estimator,
        data.data,
        y=data.target,
        cv=cv,
        return_train_score=True,
        return_estimator=True,
    )

    try:
        with NamedTemporaryFile() as tmpfile:
            joblib.dump(estimator, tmpfile.name)
            experiment.add_artifact(tmpfile.name, name='scores.joblib')
    except Exception as exception:
        warn(f'Artifact save failed: {exception}')
    experiment.log_scalar(
        'score_mean',
        np.nanmean(scores['test_score']),
    )
    experiment.log_scalar(
        'score_std',
        np.nanstd(scores['test_score']),
    )


def experiment(
    dataset: Callable[..., Bunch],
    estimator: Callable[..., BaseEstimator],
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
    dataset_ingredient = Ingredient('dataset')
    dataset = dataset_ingredient.capture(dataset)
    estimator_ingredient = Ingredient('estimator')
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
        cv = getattr(data, 'inner_cv', None)

        try:
            e = estimator(cv=cv)
        except TypeError as exception:
            warn(f'The estimator does not accept cv: {exception}')
            e = estimator()

        # Model assessment
        if getattr(data, 'test_indices', None):
            _benchmark_one(
                experiment=experiment,
                data=data,
                estimator=e,
            )
        elif getattr(data, 'outer_cv', None) is not None:
            _benchmark_partitions(
                experiment=experiment,
                data=data,
                estimator=e,
            )

    return experiment
