"""
@author: David Diaz Vico
@license: MIT
"""

from . import libsvm, raetsch, sklearn, uci, ucr
repos = {'libsvm': libsvm, 'raetsch': raetsch, 'sklearn': sklearn, 'uci': uci,
         'ucr': ucr}
try:
    from . import cran
    repos['cran'] = cran
except ImportError:
    pass
try:
    from . import forex
    repos['forex'] = forex
except ImportError:
    pass
try:
    from . import keel
    repos['keel'] = keel
except ImportError:
    pass
try:
    from . import keras
    repos['keras'] = keras
except ImportError:
    pass


def fetch(repository, dataset, collection=None, **kwargs):
    if collection:
        data = repos[repository].fetch(collection, dataset, **kwargs)
    else:
        data = repos[repository].fetch(dataset, **kwargs)
    return data
