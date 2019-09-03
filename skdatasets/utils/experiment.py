"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sacred import Experiment, Ingredient
from sklearn.model_selection import cross_validate, PredefinedSplit
from tempfile import TemporaryFile


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
    def run(return_estimator=False, save_output=False):
        """Run the experiment.

        Run the experiment.

        Parameters
        ----------
        return_estimator : boolean, default False
            Whether to return the estimator or estimators fitted.
        save_output : boolean, default False
            Whether to save the output as an artifact.

        """
        data = dataset()
        for a in ('target', 'data_test', 'target_test', 'inner_cv', 'outer_cv'):
            if a not in data:
                setattr(data, a, None)

        def _estimator(cv=None):
            """Create an estimator with or without hyperparameter search."""
            try:
                e = estimator(cv=cv)
            except:
                e = estimator()
            return e

        def _output(e, X):
            """Generate the outputs of an estimator."""
            outputs = dict()
            for output in ('transform', 'predict'):
                if hasattr(e, output):
                    outputs[output] = getattr(e, output)(X)
            return outputs

        # Inner CV for metaparameter search
        if hasattr(data.inner_cv, '__iter__'):  # Explicit CV folds
            X = np.array([]).reshape((0, *data.inner_cv[0][0].shape[1:]))
            y = np.array([]).reshape((0, *data.inner_cv[0][1].shape[1:]))
            cv = []
            for i, (X_, y_, X_test_, y_test_) in enumerate(data.inner_cv):
                X = np.concatenate((X, X_, X_test_))
                y = np.concatenate((y, y_, y_test_))
                cv = cv + [-1]*len(X_) + [i]*len(X_test_)
            e = _estimator(cv=PredefinedSplit(cv))
            e.fit(X, y=y)
            if hasattr(e, 'best_estimator_'):
                e.fit = e.best_estimator_.fit
        else:  # Automatic/indexed CV folds
            e = _estimator(cv=data.inner_cv)

        # Outer CV/test partition for model assessment
        if data.data_test is not None:  # Test partition
            e.fit(data.data, y=data.target)
            scores = {'test_score': [e.score(data.data_test,
                                             y=data.target_test)]}
            if return_estimator:
                scores['estimator'] = [e]
            if save_output:
                with TemporaryFile() as tmpfile:
                    np.save(tmpfile, _output(e, data.data_test))
                    experiment.add_artifact(tmpfile, name='output.npy')
        else:  # Outer CV
            if hasattr(data.outer_cv, '__iter__'):  # Explicit CV folds
                scores = {'test_score': []}
                if return_estimator:
                    scores['estimator'] = []
                for i, (X, y, X_test, y_test) in enumerate(data.outer_cv):
                    e.fit(X, y=y)
                    scores['test_score'].append(e.score(X_test, y=y_test))
                    if return_estimator:
                        scores['estimator'].append(e)
                    if save_output:
                        with TemporaryFile() as tmpfile:
                            np.save(tmpfile, _output(e, X_test))
                            experiment.add_artifact(tmpfile,
                                                    name=f'output_{i}.npy')
            else:  # Automatic/indexed CV folds
                scores = cross_validate(e, data.data, y=data.target,
                                        cv=data.outer_cv,
                                        return_estimator=return_estimator)
        experiment.log_scalar('score_mean', np.nanmean(scores['test_score']))
        experiment.log_scalar('score_std', np.nanstd(scores['test_score']))
        experiment.info.update(scores)

    return experiment
