"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.libsvm import (load_skin_nonskin, load_australian, load_a4a,
                               load_cod_rna, load_combined, load_dna)

from .base import load, use


def test_libsvm():
    """Tests libsvm datasets."""
    for loader in [load_skin_nonskin, load_australian, load_a4a, load_cod_rna,
                   load_combined, load_dna]:
        load(loader)
        use(loader)
