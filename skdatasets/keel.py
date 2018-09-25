"""
Keel datasets (http://sci2s.ugr.es/keel/).

@author: David Diaz Vico
@license: MIT
"""

from os.path import join

from sklearn.datasets.base import Bunch

import numpy as np
import pandas as pd

from .base import CustomSplit, fetch_file, fetch_zip


def _load_fold(fold, name, nfolds, data_home_fold, names, feature_names,
               target_name):
    """Load a dataset fold."""
    data = pd.read_csv(join(data_home_fold,
                            name + '-' + str(nfolds) + '-' + str(fold + 1) +
                            'tra.dat'),
                       names=names, sep=', ', engine='python',
                       skiprows=len(names) + 4, na_values='?')
    X = pd.get_dummies(data[feature_names]).values
    y = pd.factorize(data[target_name].tolist(), sort=True)[0]
    data_test = pd.read_csv(join(data_home_fold,
                                 name + '-' + str(nfolds) + '-' +
                                 str(fold + 1) + 'tst.dat'),
                            names=names, sep=', ', engine='python',
                            skiprows=len(names) + 4, na_values='?')
    X_test = pd.get_dummies(data_test[feature_names]).values
    y_test = pd.factorize(data_test[target_name].tolist(), sort=True)[0]
    return X, y, X_test, y_test


def _load_folds(name, nfolds, url, names, feature_names, target_name):
    """Load a dataset folds."""
    data_home_fold = fetch_zip(name,
                               url + '/' + name + '-' + str(nfolds) +
                               '-fold.zip')
    X = []
    y = []
    X_test = []
    y_test = []
    for i in range(nfolds):
        _X, _y, _X_test, _y_test = _load_fold(i, name, nfolds, data_home_fold,
                                              names, feature_names,
                                              target_name)
        X.append(_X)
        y.append(_y)
        X_test.append(_X_test)
        y_test.append(_y_test)
    return X, y, X_test, y_test


def _fetch(name, url, names, target_name, nfolds=5):
    """Fetch dataset."""
    if url == 'http://sci2s.ugr.es/keel/dataset/data/classification':
        filename_descr = fetch_file(name, url + '/' + name + '-names.txt')
    else:
        filename_descr = fetch_file(name, url + '/names/' + name +
                                    '-names.txt')
    feature_names = [n for n in names if n != target_name]
    with open(filename_descr) as rst_file:
        fdescr = rst_file.read()
    if (nfolds is None) or (nfolds == 1):
        data_home = fetch_zip(name, url + '/' + name + '.zip')
        data = pd.read_csv(join(data_home, name + '.dat'), names=names,
                           sep=',\s*', engine='python',
                           skiprows=len(names) + 4, na_values='?')
        X = pd.get_dummies(data[feature_names]).values
        y = pd.factorize(data[target_name].tolist(), sort=True)[0]
        outer_cv = None
    elif nfolds in (5, 10):
        (features_folds, target_folds,
         features_folds_test, target_folds_test) = _load_folds(name,
                                                               nfolds,
                                                               url,
                                                               names,
                                                               feature_names,
                                                               target_name)
        tr_splits = []
        tr_counter = 0
        ts_splits = []
        ts_counter = 0
        X = []
        y = []
        for ftr, ttr, fts, tts in zip(features_folds, target_folds,
                                      features_folds_test, target_folds_test):
            tr_splits.append(range(tr_counter, len(ttr) + tr_counter))
            tr_counter += len(ttr)
            X.append(ftr)
            y.append(ttr)
            ts_splits.append(range(ts_counter, len(tts) + ts_counter))
            ts_counter += len(tts)
            X.append(fts)
            y.append(tts)
        X = np.vstack(X)
        y = np.hstack(y)
        outer_cv = CustomSplit(tr_splits, ts_splits)
    return X, y, outer_cv, target_name, fdescr, feature_names


BASE_URL = 'http://sci2s.ugr.es/keel/'

datasets = {
    'abalone9-18': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Sex', 'Length', 'Diameter', 'Height',
                  'Whole_weight', 'Shucked_weight',
                  'Viscera_weight', 'Shell_weight',
                  'Class'],
                 'Class']
        },
    'abalone19': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Sex', 'Length', 'Diameter', 'Height',
                  'Whole_weight', 'Shucked_weight',
                  'Viscera_weight', 'Shell_weight',
                  'Class'],
                 'Class']
        },
    'balance': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['left-weight', 'left-distance',
                  'right-weight', 'right-distance', 'class'],
                 'class']
        },
    'cleveland': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['Age', 'Sex', 'Cp', 'Trestbps', 'Chol',
                  'Fbs', 'Restecg', 'Thalach', 'Exang',
                  'Oldpeak', 'Slope', 'Ca', 'Thal', 'Num'],
                 'Num']
        },
    'cleveland-0_vs_4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['age', 'sex', 'cp', 'trestbps',
                  'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak',
                  'slope', 'ca', 'thal', 'num'],
                 'num']
        },
    'ecoli': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1',
                  'alm2', 'class'],
                 'class']
        },
    'ecoli-0-1-3-7_vs_2-6': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Lip', 'Chg', 'Aac',
                  'Alm1', 'Alm2', 'Class'],
                 'Class']
        },
    'ecoli-0-1-4-6_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a5', 'a6',
                  'a7', 'class'],
                 'class']
        },
    'ecoli-0-1-4-7_vs_2-3-5-6': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                  'a7', 'class'],
                 'class']
        },
    'ecoli-0-1-4-7_vs_5-6': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a5', 'a6',
                  'a7', 'class'],
                 'class']
        },
    'ecoli-0-1_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a5', 'a6', 'a7',
                  'class'],
                 'class']
        },
    'ecoli-0-3-4-6_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a4', 'a5',
                  'a6', 'a7', 'class'],
                 'class']
        },
    'ecoli-0-3-4-7_vs_5-6': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a4', 'a5',
                  'a6', 'a7', 'class'],
                 'class']
        },
    'ecoli-0-6-7_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['a1', 'a2', 'a3', 'a5', 'a6', 'a7',
                  'class'],
                 'class']
        },
    'ecoli4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Lip', 'Chg', 'Aac', 'Alm1',
                  'Alm2', 'Class'],
                 'Class']
        },
    'glass': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass-0-1-4-6_vs_2': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass-0-1-6_vs_2': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass-0-1-6_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass-0-4_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass-0-6_vs_5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'typeGlass'],
                 'typeGlass']
        },
    'glass2': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'Class'],
                 'Class']
        },
    'glass4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'Class'],
                 'Class']
        },
    'glass5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba',
                  'Fe', 'Class'],
                 'Class']
        },
    'led7digit-0-2-4-5-6-7-8-9_vs_1': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p2',
                 ['Led1', 'Led2',
                  'Led3', 'Led4',
                  'Led5', 'Led6',
                  'Led7', 'number'],
                 'number']
        },
    'newthyroid': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['T3resin', 'thyroxin',
                  'triiodothyronine', 'thyroidstimulating',
                  'TSH_value', 'class'],
                 'class']
        },
    'page-blocks-1-3_vs_4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Height', 'Lenght', 'Area',
                  'Eccen', 'P_black', 'P_and',
                  'Mean_tr', 'Blackpix',
                  'Blackand', 'Wb_trans', 'Class'],
                 'Class']
        },
    'satimage': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['Sp11', 'Sp12', 'Sp13', 'Sp14', 'Sp15',
                  'Sp16', 'Sp17', 'Sp18', 'Sp19', 'Sp21',
                  'Sp22', 'Sp23', 'Sp24', 'Sp25', 'Sp26',
                  'Sp27', 'Sp28', 'Sp29', 'Sp31', 'Sp32',
                  'Sp33', 'Sp34', 'Sp35', 'Sp36', 'Sp37',
                  'Sp38', 'Sp39', 'Sp41', 'Sp42', 'Sp43',
                  'Sp44', 'Sp45', 'Sp46', 'Sp47', 'Sp48',
                  'Sp49', 'Class'],
                 'Class']
        },
    'shuttle-c0-vs-c4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                  'A7', 'A8', 'A9', 'Class'],
                 'Class']
        },
    'shuttle-c2-vs-c4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                  'A7', 'A8', 'A9', 'Class'],
                 'Class']
        },
    'vowel0': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['TT', 'SpeakerNumber', 'Sex', 'F0', 'F1',
                  'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                  'F9', 'Class'],
                 'Class']
        },
    'yeast': {
        'args': [BASE_URL +
                 'dataset/data/classification',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast-0-5-6-7-9_vs_4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast-1-2-8-9_vs_7': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast-1-4-5-8_vs_7': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast-1_vs_7': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast-2_vs_8': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast4': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast5': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        },
    'yeast6': {
        'args': [BASE_URL +
                 'keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                 ['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox',
                  'Vac', 'Nuc', 'Class'],
                 'Class']
        }
    }


def load(name, return_X_y=False, **kwargs):
    """Load

    Load a dataset.

    Parameters
    ----------
    name: string
          Dataset name.
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..
    **kwargs: dict
              Optional key-value arguments

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y, X_test, y_test, inner_cv, outer_cv: arrays
                                              If return_X_y is True

    """
    (X, y, outer_cv, target_name,
     DESCR, feature_names) = _fetch(name,
                                    *datasets[name]['args'],
                                    **kwargs)
    if return_X_y:
        return X, y, None, None, None, outer_cv
    return Bunch(data=X, target=y, outer_cv=outer_cv, target_names=target_name,
                 DESCR=DESCR, feature_names=feature_names)
