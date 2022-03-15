Repositories
============

The core of the scikit-datasets package consist in fetching functions to
obtain data from several repositories, containing both multivariate and
functional data.

The subpackage :mod:`~skdatasets.repositories` contains a module per available
repository. For repositories that contain data in a regular format, that module
has a ``fetch`` function that returns data in a
:doc:`standardized format <structure>`.
For modules such as :mod:`~skdatasets.repositories.cran`, where data is in
a non-regular format, specific functions are provided to return the data.

The available repositories are described next.

Aneurisk
--------

The Aneurisk dataset repository

URL: http://ecm2.mathcs.emory.edu/aneuriskweb/index

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.aneurisk.fetch

CRAN
----

The main repository of R packages.

URL: https://cran.r-project.org/

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.cran.fetch_package
   ~skdatasets.repositories.cran.fetch_dataset

Forex
-----

The foreign exchange market (Forex).

URL: https://theforexapi.com/

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.forex.fetch

Keel
----

The KEEL-dataset repository.

URL: https://sci2s.ugr.es/keel/datasets.php

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.keel.fetch

Keras
-----

The Keras example datasets.

URL: https://keras.io/api/datasets

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.keras.fetch

LIBSVM
------

The LIBSVM data repository.

URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.libsvm.fetch

Rätsch
-------

The Gunnar Rätsch benchmark datasets.

URL: https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets/

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.raetsch.fetch

scikit-learn
------------

The scikit-learn example datasets.

URL: https://scikit-learn.org/stable/datasets.html

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.sklearn.fetch

UCI
---

The University of California Irvine (CRAN) repository.

URL: https://archive.ics.uci.edu

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.uci.fetch

UCR
---

The UCR/UEA time series classification archive.

URL: https://www.timeseriesclassification.com

.. autosummary::
   :toctree: autosummary

   ~skdatasets.repositories.ucr.fetch
