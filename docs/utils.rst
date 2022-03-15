Utilities
=========

In addition to dataset fetching, scikit-datasets provide some utility functions
that make easier dataset-related tasks, such as launching experiments and
formatting their scores.

Estimator
---------

The following functions are related :external:term:`estimators` that follow the
scikit-learn API.

.. autosummary::
   :toctree: autosummary

   ~skdatasets.utils.estimator.json2estimator
   
Experiment
----------

The following functions are related to launching machine learning experiments.

.. autosummary::
   :toctree: autosummary

   ~skdatasets.utils.experiment.experiment

Scores
------

The following functions can be used to format and display the scores of machine
learning or hypothesis testing experiments.

.. autosummary::
   :toctree: autosummary

   ~skdatasets.utils.scores.scores_table
   ~skdatasets.utils.scores.hypotheses_table
   