"""
@author: David Diaz Vico
@license: MIT
"""

import joblib
import numpy as np
import os
from sacred import Experiment, Ingredient
from sklearn.model_selection import cross_validate, PredefinedSplit
from tempfile import mkdtemp, NamedTemporaryFile
from time import process_time
from warnings import warn


def experiment(dataset, estimator):
    """Prepare a Scikit-learn experiment as a Sacred experiment.

    Prepare a Scikit-learn experiment indicating a dataset and an estimator and
    return it as a Sacred experiment.

    Parameters
    ----------
    dataset : function
        Dataset fetch function. Might receive any argument. Must return a Bunch
        with data, target (might be None), inner_cv (might be None) and outer_cv
        (might be None).
    estimator : function
        Estimator initialization function. Might receive any keyword argument.
        Must return an initialized sklearn-compatible estimator.

    Returns
    -------
    experiment : Experiment
        Sacred experiment, ready to be run.

    """

    _dataset = Ingredient('dataset')
    dataset = _dataset.capture(dataset)
    _estimator = Ingredient('estimator')
    estimator = _estimator.capture(estimator)
    experiment = Experiment(ingredients=(_dataset, _estimator))

    @experiment.automain
    def run():
        """Run the experiment."""
        data = dataset()

        # Metaparameter search
        X = data.data
        y = data.target
        cv = None
        explicit_cv_folds = False
        if hasattr(data, 'inner_cv'):
            cv = data.inner_cv
            explicit_cv_folds = hasattr(data.inner_cv, '__iter__')
            if explicit_cv_folds:
                # Explicit CV folds
                X = np.array([]).reshape((0, *data.inner_cv[0][0].shape[1:]))
                y = np.array([]).reshape((0, *data.inner_cv[0][1].shape[1:]))
                cv = []
                for i, (X_, y_, X_test_, y_test_) in enumerate(data.inner_cv):
                    X = np.concatenate((X, X_, X_test_))
                    y = np.concatenate((y, y_, y_test_))
                    cv = cv + [-1]*len(X_) + [i]*len(X_test_)
                cv = PredefinedSplit(cv)
        try:
            e = estimator(cv=cv)
        except Exception as exception:
            warn(f'The estimator does not accept cv: {exception}')
            e = estimator()
        if explicit_cv_folds:
            e.fit(X, y=y)
            e.fit = e.best_estimator_.fit

        # Model assessment
        if hasattr(data, 'data_test') and (data.data_test is not None):
            # Test partition
            e.fit(X, y=y)
            try:
                with NamedTemporaryFile() as tmpfile:
                    joblib.dump(e, tmpfile.name)
                    experiment.add_artifact(tmpfile.name,
                                            name='estimator.joblib')
            except Exception as exception:
                warn(f'Artifact save failed: {exception}')
            experiment.log_scalar('score_mean', e.score(data.data_test,
                                  y=data.target_test))
            experiment.log_scalar('score_std', 0.0)
            for output in ('transform', 'predict'):
                if hasattr(e, output):
                    with open(os.path.join(mkdtemp(), f'{output}.npy'),
                              'wb+') as tmpfile:
                        np.save(tmpfile, getattr(e, output)(data.data_test))
                        experiment.add_artifact(tmpfile.name)
        elif hasattr(data, 'outer_cv'):
            # Outer CV
            if hasattr(data.outer_cv, '__iter__'):
                # Explicit CV folds
                scores = {'test_score': list(), 'train_score': list(),
                          'fit_time': list(), 'score_time': list(),
                          'estimator': list()}
                outputs = {'transform': list(), 'predict': list()}
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
                            outputs[output].append([getattr(e, output)(X_test)])
                for output in ('transform', 'predict'):
                    if outputs[output]:
                        with open(os.path.join(mkdtemp(), f'{output}.npy'),
                                  'wb+') as tmpfile:
                            np.save(tmpfile, np.array(outputs[output]))
                            experiment.add_artifact(tmpfile.name)
            else:
                # Automatic/indexed CV folds
                scores = cross_validate(e, data.data, y=data.target,
                                        cv=data.outer_cv,
                                        return_train_score=True,
                                        return_estimator=True)
            try:
                with NamedTemporaryFile() as tmpfile:
                    joblib.dump(e, tmpfile.name)
                    experiment.add_artifact(tmpfile.name, name='scores.joblib')
            except Exception as exception:
                warn(f'Artifact save failed: {exception}')
            experiment.log_scalar('score_mean',
                                  np.nanmean(scores['test_score']))
            experiment.log_scalar('score_std', np.nanstd(scores['test_score']))

    return experiment
