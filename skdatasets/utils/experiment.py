"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sacred import Experiment, Ingredient
from sklearn.model_selection import cross_validate, PredefinedSplit
from tempfile import NamedTemporaryFile
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
        """Run the experiment.

        Run the experiment.

        """
        data = dataset()

        # Metaparameter search
        X = data.data
        y = data.target
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
        else:
            cv = None
            explicit_cv_folds = False
        try:
            e = estimator(cv=cv)
        except Exception as exception:
            warn(str(exception))
            e = estimator()
        e.fit(X, y=y)
        with NamedTemporaryFile() as tmpfile:
            np.save(tmpfile, e)
            experiment.add_artifact(tmpfile.name, name='estimator.npy')
        items = dict()
        for item in ['cv_results_', 'best_score_', 'best_params_',
                    'best_index_', 'n_splits_', 'refit_time_']:
            if hasattr(e, item):
                items[item] = getattr(e, item)
        if items:
            experiment.info.update({'inner_cv': items})
        if explicit_cv_folds:
            e = e.best_estimator_

        # Model assessment
        if hasattr(data, 'data_test') and (data.data_test is not None):
            # Test partition
            experiment.log_scalar('score', e.score(data.data_test,
                                                   y=data.target_test))
            for output in ('transform', 'predict'):
                if hasattr(e, output):
                    with NamedTemporaryFile() as tmpfile:
                        np.save(tmpfile,
                                getattr(e, output)(data.data_test))
                        experiment.add_artifact(tmpfile.name,
                                                name=f'{output}.npy')
        elif hasattr(data, 'outer_cv'):
            # Outer CV
            if hasattr(data.outer_cv, '__iter__'):
                # Explicit CV folds
                items = dict()
                for i, (X, y, X_test, y_test) in enumerate(data.outer_cv):
                    e.fit(X, y=y)
                    with NamedTemporaryFile() as tmpfile:
                        np.save(tmpfile, e)
                        experiment.add_artifact(tmpfile.name,
                                                name=f'estimator_{i}.npy')
                    experiment.log_scalar('score', e.score(X_test, y=y_test))
                    items_i = dict()
                    for item in ['cv_results_', 'best_score_', 'best_params_',
                                'best_index_', 'n_splits_', 'refit_time_']:
                        if hasattr(e, item):
                            items_i[item] = getattr(e, item)
                    if items_i:
                        items[i] = items_i
                    for output in ('transform', 'predict'):
                        if hasattr(e, output):
                            with NamedTemporaryFile() as tmpfile:
                                np.save(tmpfile, getattr(e, output)(X_test))
                                experiment.add_artifact(tmpfile.name,
                                                        name=f'{output}_{i}.npy')
                if items:
                    experiment.info.update({'outer_cv': items})
            else:
                # Automatic/indexed CV folds
                scores = cross_validate(e, data.data, y=data.target,
                                        cv=data.outer_cv,
                                        return_train_score=True,
                                        return_estimator=True)
                with NamedTemporaryFile() as tmpfile:
                    np.save(tmpfile, scores)
                    experiment.add_artifact(tmpfile.name, name=f'scores.npy')
                experiment.log_scalar('score', scores['test_score'])
                items = dict()
                for item in ['test_score', 'train_score', 'fit_time',
                             'score_time']:
                    items[item] = scores[item]
                estimators = dict()
                for i, e in enumerate(scores['estimator']):
                    with NamedTemporaryFile() as tmpfile:
                        np.save(tmpfile, e)
                        experiment.add_artifact(tmpfile.name,
                                                name=f'estimator_{i}.npy')
                    items_i = dict()
                    for item in ['cv_results_', 'best_score_', 'best_params_',
                                'best_index_', 'n_splits_', 'refit_time_']:
                        if hasattr(e, item):
                            items_i[item] = getattr(e, item)
                    if items_i:
                        estimators[i] = items_i
                items['estimator'] = estimators
                experiment.info.update({'outer_cv': items})

    return experiment
