"""
Scikit-learn-compatible datasets.

@author: David Diaz Vico
@license: MIT
"""

from . import libsvm, raetsch, sklearn, uci, ucr
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
