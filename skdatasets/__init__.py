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
except Exception:
    pass

try:
    from . import cran
    repositories.append(cran)
except ImportError:
    pass

for repository in repositories:
    for dataset in repository.datasets.keys():
        setattr(repository,
                'load_' + dataset.replace('-', '_').replace('.', '_'),
                partial(repository.load, dataset))

loader = {'gunnar_raetsch': gunnar_raetsch.load, 'keel': keel.load,
          'libsvm': libsvm.load, 'sklearn': sklearn.load, 'uci': uci.load}
try:
    loader.update({'keras': keras.load})
except:
    pass

try:
    loader.update({'cran': cran.load})
except:
    pass


def load(repository, dataset, **kwargs):
    """ Select a dataset. """
    return loader[repository](dataset, **kwargs)
