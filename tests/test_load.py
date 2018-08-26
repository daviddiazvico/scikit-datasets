"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import check_items

from skdatasets import load


def test_load():
    """Tests load."""
    X, y, X_test, y_test, inner_cv, outer_cv = load('gunnar_raetsch', 'banana')
    check_items([X, y, outer_cv], [X_test, y_test, inner_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('keel', 'abalone9-18')
    check_items([X, y, outer_cv], [X_test, y_test, inner_cv])
    try:
        X, y, X_test, y_test, inner_cv, outer_cv = load('keras', 'mnist')
        check_items([X, y, X_test, y_test], [inner_cv, outer_cv])
    except:
        pass
    X, y, X_test, y_test, inner_cv, outer_cv = load('libsvm', 'a4a')
    check_items([X, y, X_test, y_test], [inner_cv, outer_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('libsvm', 'dna')
    check_items([X, y, X_test, y_test, inner_cv], [outer_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('libsvm', 'abalone')
    check_items([X, y], [X_test, y_test, inner_cv, outer_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('sklearn', 'iris')
    check_items([X, y], [X_test, y_test, inner_cv, outer_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('uci', 'abalone')
    check_items([X, y], [X_test, y_test, inner_cv, outer_cv])
    X, y, X_test, y_test, inner_cv, outer_cv = load('uci', 'adult')
    check_items([X, y, X_test, y_test], [inner_cv, outer_cv])
