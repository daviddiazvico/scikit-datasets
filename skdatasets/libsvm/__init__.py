"""
LIBSVM datasets (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

@author: David Diaz Vico
@license: MIT
"""

from .classification import load_skin_nonskin
from .classification_scale import (load_australian, load_covtype_binary,
                                   load_diabetes, load_german_numer, load_heart)
from .classification_test import (load_a4a, load_a8a, load_pendigits, load_usps,
                                  load_w7a, load_w8a)
from .classification_test_remaining import load_cod_rna
from .classification_test_scale import load_combined, load_news20
from .classification_val_test import (load_dna, load_ijcnn1, load_letter,
                                      load_satimage, load_shuttle)
from .regression import (load_abalone, load_bodyfat, load_cpusmall,
                         load_housing, load_mg, load_mpg, load_pyrim,
                         load_space_ga, load_triazines)
