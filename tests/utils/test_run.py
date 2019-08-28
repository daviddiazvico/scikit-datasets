"""
@author: David Diaz Vico
@license: MIT
"""

import subprocess


def test_binary_classification():
    """Tests binary classification experiment."""
    subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'binary', '-d', 'breast-cancer', '-e', 'tests/utils/MLPClassifier.json'])
    

def test_multiclass_classification():
    """Tests multiclass classification experiment."""
    subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'multiclass', '-d', 'iris', '-e', 'tests/utils/LinearRegression.json'])
    

def test_regression():
    """Tests regression experiment."""
    subprocess.call(['skdatasets/utils/run.py', '-r', 'libsvm', '-c', 'regression', '-d', 'housing', '-e', 'tests/utils/MLPRegressor.json'])
    