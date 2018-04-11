"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

@author: David Diaz Vico
@license: MIT
"""

from .classification import load_skin_nonskin
from .classification_scale import (load_australian, load_covtype_binary,
                                   load_diabetes, load_german_numer, load_heart)
from .classification_test import (load_a4a, load_a8a, load_epsilon,
                                  load_pendigits, load_usps, load_w7a, load_w8a)
from .classification_test_remaining import load_cod_rna
from .classification_test_scale import load_combined, load_news20
from .classification_val_test import (load_dna, load_ijcnn1, load_letter,
                                      load_satimage, load_shuttle)
from .regression import (load_abalone, load_bodyfat, load_cpusmall,
                         load_housing, load_mg, load_mpg, load_pyrim,
                         load_space_ga, load_triazines)


load = {'skin_nonskin': load_skin_nonskin, 'australian': load_australian,
        'covtype.binary': load_covtype_binary, 'diabetes': load_diabetes,
        'german.numer': load_german_numer, 'heart': load_heart,
        'a4a': load_a4a, 'a8a': load_a8a, 'epsilon': load_epsilon,
        'pendigits': load_pendigits, 'usps': load_usps, 'w7a': load_w7a,
        'w8a': load_w8a, 'cod-rna': load_cod_rna, 'combined': load_combined,
        'news20': load_news20, 'dna': load_dna, 'ijcnn1': load_ijcnn1,
        'letter': load_letter, 'satimage': load_satimage,
        'shuttle': load_shuttle, 'abalone': load_abalone,
        'bodyfat': load_bodyfat, 'cpusmall': load_cpusmall,
        'housing': load_housing, 'mg': load_mg, 'mpg': load_mpg,
        'pyrim': load_pyrim, 'space_ga': load_space_ga,
        'triazines': load_triazines}
