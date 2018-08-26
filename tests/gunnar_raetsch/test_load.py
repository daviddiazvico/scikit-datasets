"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ..base import check_load_dataset

from skdatasets.gunnar_raetsch import (load_banana, load_breast_cancer,
                                       load_diabetis, load_flare_solar,
                                       load_german, load_heart, load_image,
                                       load_ringnorm, load_splice, load_thyroid,
                                       load_titanic, load_twonorm,
                                       load_waveform)


def test_split():
    """Tests Gunnar Raetsch dataset splits."""
    data = load_banana()
    X = data.features
    y = data.target
    splits = data.splits
    cross_val_score(LogisticRegression(), X, y=y, cv=splits)


def test_load():
    """Tests Gunnar Raetsch benchmark datasets."""
    datasets = {'banana': {'loader': load_banana, 'n_patterns': (5300, ),
                           'n_variables': 2},
                'breast_cancer': {'loader': load_breast_cancer,
                                  'n_patterns': (263, ), 'n_variables': 9},
                'diabetis': {'loader': load_diabetis, 'n_patterns': (768, ),
                             'n_variables': 8},
                'flare_solar': {'loader': load_flare_solar,
                                'n_patterns': (144, ), 'n_variables': 9},
                'german': {'loader': load_german, 'n_patterns': (1000, ),
                           'n_variables': 20},
                'heart': {'loader': load_heart, 'n_patterns': (270, ),
                          'n_variables': 13},
                'image': {'loader': load_image, 'n_patterns': (2086, ),
                          'n_variables': 18},
                'ringnorm': {'loader': load_ringnorm, 'n_patterns': (7400, ),
                             'n_variables': 20},
                'splice': {'loader': load_splice, 'n_patterns': (2991, ),
                           'n_variables': 60},
                'thyroid': {'loader': load_thyroid, 'n_patterns': (215, ),
                            'n_variables': 5},
                'titanic': {'loader': load_titanic, 'n_patterns': (24, ),
                            'n_variables': 3},
                'twonorm': {'loader': load_twonorm, 'n_patterns': (7400, ),
                            'n_variables': 20},
                'waveform': {'loader': load_waveform, 'n_patterns': (5000, ),
                             'n_variables': 21}}
    for dataset in datasets.values():
        check_load_dataset(dataset['loader'], dataset['n_patterns'],
                           dataset['n_variables'], (('features', 'target'), ),
                           n_targets=1, n_folds=None)
