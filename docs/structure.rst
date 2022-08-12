Dataset structure
=================

Most of the repositories available in scikit-datasets have datasets in some
regular format.
In that case, its corresponding ``fetch`` function in scikit-datasets converts
the data to a standardized format, similar to the one used in scikit-learn, but
with new optional fields for additional features that some repositories
include, such as indices for train, validation and test partitions.

.. note::
	Data in the CRAN repository is unstructured, and thus there is no ``fetch``
	function for it. The data is returned in the original format.

The structure is a :external:class:`~sklearn.utils.Bunch` object with the
following fields:

- ``data``: The matrix of observed data. A 2d NumPy array, ready to be used
  with scikit-learn tools.
  Each row correspond to a different observation while each column is a
  particular feature.
  For datasets with train, validation and test partitions, the whole data
  is included here.
  Use ``train_indices``, ``validation_indices`` and ``test_indices`` to
  select each partition.
- ``target``: The target of the classification or regression problem. This
  is a 1d NumPy array except for multioutput problems, in with it is a 2d
  array, where each column correspond to a different output.
- ``DESCR``: A human readable description of the dataset.
- ``feature_names``: The list of feature names, if the repository has that
  information available.
- ``target_names``: For classification problems, this correspond to the names
  of the different classes, if available.
  Note that this field in scikit-learn is used in some cases for naming the
  outputs in multioutput problems.
  As we will try to maintain compatibility with scikit-learn, the meaning of
  this field could change in future versions.
- ``train_indices``: Indexes of the elements of the train partition, if
  available in the repository.
- ``validation_indices``: Indexes of the elements of the validation partition,
  if available in the repository.
- ``test_indices``: Indexes of the elements of the test partition, if
  available in the repository.
- ``inner_cv``: A :external:term:`CV splitter` object, usable for cross
  validation and hyperparameter selection, if the repository provides a
  cross validation strategy (such as using a particular validation
  partition).
- ``outer_cv``: A Python iterable over different train and test partitions,
  when they are provided in the repository.
