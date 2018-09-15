"""
Scikit-learn-compatible datasets.

@author: David Diaz Vico
@license: MIT
"""

from functools import partial

from . import gunnar_raetsch, libsvm, keel, sklearn, uci
repositories = [gunnar_raetsch, libsvm, keel, sklearn, uci]
try:
    from . import keras
    repositories.append(keras)
except:
    pass

for repository in repositories:
    for dataset in repository.datasets.keys():
        setattr(repository,
                'load_' + dataset.replace('-', '_').replace('.', '_'),
                partial(repository.load, dataset))
