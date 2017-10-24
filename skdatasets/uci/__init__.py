"""
UCI datasets (https://archive.ics.uci.edu/ml/datasets.html).

@author: David Diaz Vico
@license: MIT
"""

from .classification import (load_abalone, load_nursery,
                             load_pima_indians_diabetes)
from .classification_test.adult import load_adult
