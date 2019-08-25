"""
@author: David Diaz Vico
@license: MIT
"""

import numpy as np
from sacred import Experiment, Ingredient
from sklearn.model_selection import cross_validate, PredefinedSplit

from .validation import scatter_plot, metaparameter_plot, history_plot


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
    def run(cross_validate=cross_validate, return_estimator=False):
        """Run the experiment.

        Run the experiment.

        Parameters
        ----------
        cross_validate : function, default=cross_validate
            Function to evaluate metrics by cross-validation. Must receive the
            estimator, X, y (migth be None) and cv (migth be None). Must return
            a dictionary with the cross-validation score and maybe other info,
            like a list of fitted estimators.
        return_estimator : boolean, default False
            Whether to return the estimator or estimators fitted.

        """
        data = dataset()
        for a in ('target', 'data_test', 'target_test', 'inner_cv', 'outer_cv'):
            if a not in data:
                setattr(data, a, None)

        def _explicit_folds(data):
            """Prepare a dataset where the CV folds are explicit."""
            X = np.array([]).reshape((0, *data.inner_cv[0][0].shape[1:]))
            y = np.array([]).reshape((0, *data.inner_cv[0][1].shape[1:]))
            cv = []
            for i, (X_, y_, X_test_, y_test_) in enumerate(data.inner_cv):
                X = np.concatenate((X, X_, X_test_))
                y = np.concatenate((y, y_, y_test_))
                cv = cv + [-1]*len(X_) + [i]*len(X_test_)
            return X, y, cv

        def _estimator(cv=None):
            """Create an estimator with or without hyperparameter search."""
            try:
                e = estimator(cv=cv)
            except:
                e = estimator()
            return e

        def _plots(e, i, X, y):
            """Create different descriptive plots."""
            # Metaparameter plots
            image_files = metaparameter_plot(e, image_file=f'metaparameter_{i}.pdf')
            for image_file in image_files:
                experiment.add_artifact(image_file)
            # Scatter plots
            image_files = scatter_plot(X, y, e, image_file=f'scatter_{i}.pdf')
            for image_file in image_files:
                experiment.add_artifact(image_file)


        # Inner CV for metaparameter search
        if hasattr(data.inner_cv, '__iter__'):
            # Explicit CV folds
            X, y, cv = _explicit_folds(data)
            e = _estimator(cv=PredefinedSplit(cv))
            e.fit(X, y=y)
            if hasattr(e, 'best_estimator_'):
                e.fit = e.best_estimator_.fit
        else:
            # Automatic/indexed CV folds
            e = _estimator(cv=data.inner_cv)

        # Outer CV/test partition for model assessment
        if data.data_test is not None:
            # Test partition
            e.fit(data.data, y=data.target)
            scores = {'test_score': e.score(data.data_test, y=data.target_test)}
            if return_estimator:
                scores['estimator'] = [e]
            _plots(e, 0, data.data_test, data.target_test)
        else:
            # Outer CV
            if hasattr(data.outer_cv, '__iter__'):
                # Explicit CV folds
                scores = {'test_score': []}
                if return_estimator:
                    scores['estimator'] = []
                for i, (X, y, X_test, y_test) in enumerate(data.outer_cv):
                    e.fit(X, y=y)
                    scores['test_score'].append(e.score(X_test, y=y_test))
                    if return_estimator:
                        scores['estimator'].append(e)
                    _plots(e, i, X_test, y_test)
            else:
                # Automatic/indexed CV folds
                scores = cross_validate(e, data.data, y=data.target,
                                        cv=data.outer_cv,
                                        return_estimator=True)
                for i, e in enumerate(scores['estimator']):
                    _plots(e, i, data.data, data.target)
                if not return_estimator:
                    scores.pop('estimator')
        experiment.info.update(scores)

    return experiment
