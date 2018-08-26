"""
Scikit-learn-compatible datasets.

@author: David Diaz Vico
@license: MIT
"""

try:
    from keras.datasets import (boston_housing, cifar10, cifar100, fashion_mnist,
                                imdb, mnist, reuters)
except:
    pass
import numpy as np
from sklearn.datasets import (load_boston, load_breast_cancer, load_diabetes,
                              load_digits, load_iris, load_linnerud, load_wine)
from sklearn.model_selection import PredefinedSplit

from .gunnar_raetsch import (load_banana, load_breast_cancer, load_diabetis,
                             load_flare_solar, load_german, load_heart,
                             load_image, load_ringnorm, load_splice,
                             load_thyroid, load_titanic, load_twonorm,
                             load_waveform)
from .keel.imbalanced import (load_abalone9_18, load_abalone19,
                              load_cleveland_0_vs_4, load_ecoli4,
                              load_ecoli_0_1_3_7_vs_2_6,
                              load_ecoli_0_1_4_6_vs_5,
                              load_ecoli_0_1_4_7_vs_2_3_5_6,
                              load_ecoli_0_1_4_7_vs_5_6, load_ecoli_0_1_vs_5,
                              load_ecoli_0_3_4_6_vs_5,
                              load_ecoli_0_3_4_7_vs_5_6, load_ecoli_0_6_7_vs_5,
                              load_glass2, load_glass4, load_glass5,
                              load_glass_0_1_4_6_vs_2, load_glass_0_1_6_vs_2,
                              load_glass_0_1_6_vs_5, load_glass_0_4_vs_5,
                              load_glass_0_6_vs_5,
                              load_led7digit_0_2_4_5_6_7_8_9_vs_1,
                              load_page_blocks_1_3_vs_4, load_shuttle_c0_vs_c4,
                              load_shuttle_c2_vs_c4, load_vowel0, load_yeast4,
                              load_yeast5, load_yeast6,
                              load_yeast_0_5_6_7_9_vs_4,
                              load_yeast_1_2_8_9_vs_7, load_yeast_1_4_5_8_vs_7,
                              load_yeast_1_vs_7, load_yeast_2_vs_8)
from .keel.standard_classification import (load_balance, load_cleveland,
                                           load_ecoli, load_glass,
                                           load_newthyroid, load_satimage,
                                           load_yeast)
from .libsvm.classification import load_skin_nonskin
from .libsvm.classification_scale import (load_australian, load_covtype_binary,
                                          load_diabetes, load_german_numer,
                                          load_heart)
from .libsvm.classification_test import (load_a4a, load_a8a, load_epsilon,
                                         load_pendigits, load_usps, load_w7a,
                                         load_w8a)
from .libsvm.classification_test_remaining import load_cod_rna
from .libsvm.classification_test_scale import load_combined, load_news20
from .libsvm.classification_val_test import (load_dna, load_ijcnn1, load_letter,
                                             load_satimage, load_shuttle)
from .libsvm.regression import (load_abalone, load_bodyfat, load_cpusmall,
                                load_housing, load_mg, load_mpg, load_pyrim,
                                load_space_ga, load_triazines)
from .uci.classification import (load_abalone, load_nursery,
                                 load_pima_indians_diabetes)
from .uci.classification_test.adult import load_adult


loader = dict()

loader['gunnar_raetsch'] = {'banana': load_banana,
                            'breast_cancer': load_breast_cancer,
                            'diabetis': load_diabetis,
                            'flare_solar': load_flare_solar,
                            'german': load_german, 'heart': load_heart,
                            'image': load_image, 'ringnorm': load_ringnorm,
                            'splice': load_splice, 'thyroid': load_thyroid,
                            'titanic': load_titanic, 'twonorm': load_twonorm,
                            'waveform': load_waveform}


loader['keel'] = {'abalone9-18': load_abalone9_18, 'abalone19': load_abalone19,
                  'cleveland-0_vs_4': load_cleveland_0_vs_4,
                  'ecoli4': load_ecoli4,
                  'ecoli-0-1-3-7_vs_2-6': load_ecoli_0_1_3_7_vs_2_6,
                  'ecoli-0-1-4-6_vs_5': load_ecoli_0_1_4_6_vs_5,
                  'ecoli-0-1-4-7_vs_2-3-5-6': load_ecoli_0_1_4_7_vs_2_3_5_6,
                  'ecoli-0-1-4-7_vs_5-6': load_ecoli_0_1_4_7_vs_5_6,
                  'ecoli-0-1_vs_5': load_ecoli_0_1_vs_5,
                  'ecoli-0-3-4-6_vs_5': load_ecoli_0_3_4_6_vs_5,
                  'ecoli-0-3-4-7_vs_5-6': load_ecoli_0_3_4_7_vs_5_6,
                  'ecoli-0-6-7_vs_5': load_ecoli_0_6_7_vs_5,
                  'glass2': load_glass2, 'glass4': load_glass4,
                  'glass5': load_glass5,
                  'glass-0-1-4-6_vs_2': load_glass_0_1_4_6_vs_2,
                  'glass-0-1-6_vs_2': load_glass_0_1_6_vs_2,
                  'glass-0-1-6_vs_5': load_glass_0_1_6_vs_5,
                  'glass-0-4_vs_5': load_glass_0_4_vs_5,
                  'glass-0-6_vs_5': load_glass_0_6_vs_5,
                  'led7digit-0-2-4-5-6-7-8-9_vs_1': load_led7digit_0_2_4_5_6_7_8_9_vs_1,
                  'page-blocks-1-3_vs_4': load_page_blocks_1_3_vs_4,
                  'shuttle-c0-vs-c4': load_shuttle_c0_vs_c4,
                  'shuttle-c2-vs-c4': load_shuttle_c2_vs_c4,
                  'vowel0': load_vowel0, 'yeast4': load_yeast4,
                  'yeast5': load_yeast5, 'yeast6': load_yeast6,
                  'yeast-0-5-6-7-9_vs_4': load_yeast_0_5_6_7_9_vs_4,
                  'yeast-1-2-8-9_vs_7': load_yeast_1_2_8_9_vs_7,
                  'yeast-1-4-5-8_vs_7': load_yeast_1_4_5_8_vs_7,
                  'yeast-1_vs_7': load_yeast_1_vs_7,
                  'yeast-2_vs_8': load_yeast_2_vs_8, 'balance': load_balance,
                  'cleveland': load_cleveland, 'ecoli': load_ecoli,
                  'glass': load_glass, 'newthyroid': load_newthyroid,
                  'satimage': load_satimage, 'yeast': load_yeast}

try:
    loader['keras'] = {'boston_housing': boston_housing.load_data,
                       'cifar10': cifar10.load_data,
                       'cifar100': cifar100.load_data,
                       'fashion_mnist': fashion_mnist.load_data,
                       'imdb': imdb.load_data, 'mnist': mnist.load_data,
                       'reuters': reuters.load_data}
except:
    pass


loader['libsvm'] = {'skin_nonskin': load_skin_nonskin,
                    'australian': load_australian,
                    'covtype.binary': load_covtype_binary,
                    'diabetes': load_diabetes,
                    'german.numer': load_german_numer, 'heart': load_heart,
                    'a4a': load_a4a, 'a8a': load_a8a, 'epsilon': load_epsilon,
                    'pendigits': load_pendigits, 'usps': load_usps,
                    'w7a': load_w7a, 'w8a': load_w8a, 'cod-rna': load_cod_rna,
                    'combined': load_combined, 'news20': load_news20,
                    'dna': load_dna, 'ijcnn1': load_ijcnn1,
                    'letter': load_letter, 'satimage': load_satimage,
                    'shuttle': load_shuttle, 'abalone': load_abalone,
                    'bodyfat': load_bodyfat, 'cpusmall': load_cpusmall,
                    'housing': load_housing, 'mg': load_mg, 'mpg': load_mpg,
                    'pyrim': load_pyrim, 'space_ga': load_space_ga,
                    'triazines': load_triazines}


loader['sklearn'] = {'boston': load_boston, 'breast_cancer': load_breast_cancer,
                     'diabetes': load_diabetes, 'digits': load_digits,
                     'iris': load_iris, 'linnerud': load_linnerud,
                     'wine': load_wine}


loader['uci'] = {'abalone': load_abalone, 'nursery': load_nursery,
                 'pima-indians-diabetes': load_pima_indians_diabetes,
                 'adult': load_adult}


def load(collection, name):
    """ Load a dataset. """
    if collection == 'gunnar_raetsch':
        data = loader['gunnar_raetsch'][name]()
        X = data.features
        y = data.target
        X_test = y_test = inner_cv = None
        outer_cv = data.splits
    elif collection == 'keel':
        data = loader['keel'][name]()
        X = np.vstack(data.data5_test)
        y = np.hstack(data.target5_test)
        X_test = y_test = inner_cv = None
        outer_cv = PredefinedSplit([item for sublist in [[i]*len(fold) for i, fold in enumerate(data.data5_test)] for item in sublist])
    elif collection == 'keras':
        (X, y), (X_test, y_test) = loader['keras'][name]()
        if name in ['cifar10', 'cifar100', 'fashion_mnist', 'mnist']:
            X = X.reshape([X.shape[0], np.prod(X.shape[1:])]) / 256.0
            X_test = X_test.reshape([X_test.shape[0], np.prod(X_test.shape[1:])]) / 256.0
        y = y.flatten()
        y_test = y_test.flatten()
        inner_cv = outer_cv = None
    elif collection == 'libsvm':
        if name in ['a4a', 'a8a', 'cod-rna', 'combined', 'epsilon', 'news20',
                    'pendigits', 'usps', 'w7a', 'w8a']:
            (X, y), (X_test, y_test) = loader['libsvm'][name](return_X_y=True)
            y_test[y_test == -1] = 0
            inner_cv = None
        elif name in ['dna', 'ijcnn1', 'letter', 'satimage', 'shuttle']:
            (X, y), (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = loader['libsvm'][name](return_X_y=True)
            y_test[y_test == -1] = 0
            inner_cv = PredefinedSplit( [item for sublist in [[-1] * len(X_tr), [0] * len(X_val)] for item in sublist])
        elif name in ['abalone', 'australian', 'bodyfat', 'covtype.binary',
                      'cpusmall', 'diabetes', 'german_numer', 'heart',
                      'housing', 'mg', 'mpg', 'pyrim', 'skin_nonskin',
                      'space_ga', 'triazines']:
            X, y = loader['libsvm'][name](return_X_y=True)
            X_test = y_test = inner_cv = None
        y[y == -1] = 0
        outer_cv = None
    elif collection == 'sklearn':
        X, y = loader['sklearn'][name](return_X_y=True)
        X_test = y_test = inner_cv = outer_cv = None
    elif collection == 'uci':
        if name in ['abalone', 'nursery', 'pima_indians_diabetes']:
            X, y = loader['uci'][name](return_X_y=True)
            X_test = y_test = None
        elif name in ['adult']:
            (X, y), (X_test, y_test) = loader['uci'][name](return_X_y=True)
        inner_cv = outer_cv = None
    else:
        raise Exception('Dataset not available')
    return X, y, X_test, y_test, inner_cv, outer_cv
