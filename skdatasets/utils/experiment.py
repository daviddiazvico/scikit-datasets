"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import os
from tempfile import NamedTemporaryFile, mkdtemp
from time import process_time
from typing import Callable, List, Mapping
from warnings import warn

import joblib
import numpy as np
from sacred import Experiment, Ingredient
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.utils import Bunch


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
        X = data.data
        y = data.target
        cv = getattr(data, 'inner_cv', None)

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

        try:
            e = estimator(cv=cv)
        except TypeError as exception:
            warn(f'The estimator does not accept cv: {exception}')
            e = estimator()

        if cv is not None:
            e.fit(X_train_val, y_train_val)
            e.fit = e.best_estimator_.fit

        # Model assessment
        if test_indices:
            # Test partition
            e.fit(X_train_val, y_train_val)
            try:
                with NamedTemporaryFile() as tmpfile:
                    joblib.dump(e, tmpfile.name)
                    experiment.add_artifact(
                        tmpfile.name,
                        name='estimator.joblib',
                    )
            except Exception as exception:
                warn(f'Artifact save failed: {exception}')
            experiment.log_scalar(
                'score_mean',
                e.score(X_test, y_test),
            )
            experiment.log_scalar('score_std', 0.0)
            for output in ('transform', 'predict'):
                if hasattr(e, output):
                    with open(
                        os.path.join(
                            mkdtemp(),
                            f'{output}.npy',
                        ),
                        'wb+',
                    ) as tmpfile:
                        np.save(tmpfile, getattr(e, output)(X_test))
                        experiment.add_artifact(tmpfile.name)
        elif hasattr(data, 'outer_cv'):
            # Outer CV
            # Explicit CV folds
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
            for X, y, X_test, y_test in data.outer_cv:
                t0 = process_time()
                e.fit(X, y=y)
                t1 = process_time()
                test_score = e.score(X_test, y=y_test)
                t2 = process_time()
                scores['test_score'].append(test_score)
                scores['train_score'].append(e.score(X, y=y))
                scores['fit_time'].append(t1 - t0)
                scores['score_time'].append(t2 - t1)
                scores['estimator'].append(e)
                for output in ('transform', 'predict'):
                    if hasattr(e, output):
                        outputs[output].append(
                            getattr(e, output)(X_test),
                        )
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

            try:
                with NamedTemporaryFile() as tmpfile:
                    joblib.dump(e, tmpfile.name)
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

    return experiment
