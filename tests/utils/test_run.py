"""
@author: David Diaz Vico
@license: MIT
"""

import subprocess


def test_binary_classification():
    """Tests binary classification experiment."""
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'keel', '-c', 'imbalanced', '-d', 'abalone9-18', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'binary', '-d', 'breast-cancer', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'raetsch', '-d', 'banana', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0


def test_multiclass_classification():
    """Tests multiclass classification experiment."""
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'sklearn', '-d', 'iris', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'uci', '-d', 'wine', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'multiclass', '-d', 'shuttle', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'multiclass', '-d', 'usps', '-e', 'tests/utils/MLPClassifier.json'])
    assert ret == 0


def test_regression():
    """Tests regression experiment."""
    ret = subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'regression', '-d', 'housing', '-e', 'tests/utils/MLPRegressor.json'])
    assert ret == 0
