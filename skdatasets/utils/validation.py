"""
Scikit-learn-compatible visualizations for model validation.

@author: David Diaz Vico
@license: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils.multiclass import type_of_target


sns.set(style="white", palette="muted", color_codes=True)


def scatter_plot(X, y, estimator, image_file='scatter.pdf', max_features=10,
                 max_data=200, **kwargs):
    """ Scatter plot.

        Scatter plot of the transformations or/and predictions of the estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, features_shape)
           Input data.
        y : numpy array of shape [n_samples]
           Target values.
        estimator : estimator
            Fitted sklearn Transformer/Predictor object.
        image_file: string, default=...
            ...
        max_features : integer, default=10
            Maximum number of features to use in the plot
        max_data : integer, default=200
            Maximum number of data to use in the plot
        **kwargs : optional savefig named args

        Returns
        -------
        List of image filenames.
    """
    image_files = list()
    if hasattr(estimator, 'transform'):
        # Transformer
        plt.figure()
        transfs = estimator.transform(X)
        X = X[:max_data, :max_features]
        transfs = transfs[:max_data, :max_features]
        y = y[:max_data]
        target_type = type_of_target(y)
        if target_type in ('binary', 'multiclass'):
            # Classification/clustering
            names = list(range(transfs.shape[1]))
            names.append('class')
            data = pd.DataFrame(data=np.append(transfs,
                                               np.reshape(y, (len(y), 1)),
                                               axis=1),
                                columns=names)
            sns.set()
            sns.pairplot(data, hue='class', x_vars=names[:-1],
                         y_vars=names[:-1])
        elif target_type == 'continuous':
            # Regression
            names = list(range(transfs.shape[1]))
            names.append('y')
            data = pd.DataFrame(data=np.append(transfs,
                                               np.reshape(y, (-1, 1)),
                                               axis=1),
                                columns=names)
            sns.set()
            sns.pairplot(data, hue='y', x_vars=names[:-1],
                         y_vars=names[:-1])
            pass
        transformer_image_file = 'transformer_' + image_file
        plt.savefig(transformer_image_file, **kwargs)
        image_files.append(transformer_image_file)
    if hasattr(estimator, 'predict'):
        # Predictor
        plt.figure()
        preds = estimator.predict(X)
        X = X[:max_data, :max_features]
        preds = preds[:max_data]
        y = y[:max_data]
        target_type = type_of_target(y)
        if target_type in ('binary', 'multiclass'):
            # Classification/clustering
            names = list(range(X.shape[1]))
            names.append('class')
            diffs = y
            diffs[(y - preds) != 0] = -1
            data = pd.DataFrame(data=np.hstack((X, np.reshape(diffs, (-1, 1)))),
                                columns=names)
            sns.set()
            sns.pairplot(data, hue='class', x_vars=names[:-1],
                         y_vars=names[:-1])
        elif target_type == 'continuous':
            # Regression
            data = pd.DataFrame(data=np.hstack((np.reshape(y, (-1, 1)),
                                                np.reshape(preds, (-1, 1)),
                                                np.reshape(y - preds, (-1, 1)))),
                                columns=('y', 'preds', 'error'))
            sns.set()
            sns.scatterplot(x='y', y='preds', hue='error', data=data)
        predictor_image_file = 'predictor_' + image_file
        plt.savefig(predictor_image_file, **kwargs)
        image_files.append(predictor_image_file)
    return image_files


def metaparameter_plot(estimator, image_file='metaparameter.pdf', **kwargs):
    """ Metaparameter plot.

        Train and test metric plotted along a meta-parameter search space.

        Parameters
        ----------
        estimator : estimator
            Fitted sklearn SearchCV object.
        image_file: string, default=...
            ...
        **kwargs : optional savefig named args

        Returns
        -------
        List of image filenames.
    """
    image_files = list()
    if hasattr(estimator, 'cv_results_'):
        for k, v in estimator.cv_results_.items():
            if k[:6] == 'param_':
                try:
                    param_range = v.data.astype('float32')
                except:
                    continue
                test_mean = estimator.cv_results_['mean_test_score']
                test_std = estimator.cv_results_['std_test_score']
                try:
                    train_mean = estimator.cv_results_['mean_train_score']
                    train_std = estimator.cv_results_['std_train_score']
                except:
                    pass
                plt.figure()
                plt.autoscale(enable=True, axis='x')
                plt.xlabel(k)
                plt.ylabel('score')
                plt.plot(param_range, test_mean, 'o', label='Test', color='g')
                plt.fill_between(param_range, test_mean - test_std,
                                 test_mean + test_std, alpha=0.2, color='g')
                plt.plot(param_range[estimator.best_index_],
                         test_mean[estimator.best_index_], 'o', label='Best',
                         color='r')
                try:
                    plt.plot(param_range, train_mean, 'o', label='Train',
                             color='b')
                    plt.fill_between(param_range, train_mean - train_std,
                                     train_mean + train_std, alpha=0.2,
                                     color='b')
                    plt.plot(param_range[estimator.best_index_],
                             train_mean[estimator.best_index_], 'o', color='r')
                except:
                    pass
                plt.axvline(x=param_range[estimator.best_index_], color='r')
                plt.legend(loc='best')
                image_file = k + '_' + image_file
                plt.savefig(image_file, **kwargs)
                image_files.append(image_file)
    return image_files


def history_plot(history, image_file='history.pdf', **kwargs):
    """ History plot.

        Loss plotted for each training epoch.

        Parameters
        ----------
        history : history object
            Keras-like history object returned from fit.
        image_file: string, default=...
            ...
        **kwargs : optional savefig named args

        Returns
        -------
        None.
    """
    image_file = None
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc='best')
    plt.savefig(image_file, **kwargs)
    return image_file
