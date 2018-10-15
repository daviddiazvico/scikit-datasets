"""
Scikit-learn-compatible datasets.

@author: David Diaz Vico
@license: MIT
"""

from . import libsvm, raetsch, sklearn, uci
try:
    from . import cran
except ImportError:
    pass
try:
    from . import forex
except ImportError:
    pass
try:
    from . import keel
except ImportError:
    pass
try:
    from . import keras
except ImportError:
    pass


fetcher = {'libsvm': libsvm.fetch_libsvm, 'raetsch': raetsch.fetch_raetsch,
           'sklearn': sklearn.fetch_sklearn, 'uci': uci.fetch_uci}
try:
    fetcher['cran'] = cran.fetch_cran
except:
    pass
try:
    fetcher['forex'] = forex.fetch_forex
except:
    pass
try:
    fetcher['keel'] = keel.fetch_keel
except:
    pass
try:
    fetcher['keras'] = keras.fetch_keras
except:
    pass
def fetch(repository, *args, **kwargs):
    """ Select a dataset. """
    return fetcher[repository](*args, **kwargs)
