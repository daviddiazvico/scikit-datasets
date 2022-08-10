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

The following functions can be used to execute several experiments,
such as classification or regression tasks, with different datasets
for a posterior comparison.
These experiments are created using the Sacred library, storing the
most common parameters of interest, such as time required for training or
final scores.
After the experiments have finished, the final scores can be easily 
retrieved in order to plot a table or perform hypothesis testing.

.. autosummary::
   :toctree: autosummary

   ~skdatasets.utils.experiment.create_experiments
   ~skdatasets.utils.experiment.run_experiments
   ~skdatasets.utils.experiment.fetch_scores
   ~skdatasets.utils.experiment.ScoresInfo

Scores
------

The following functions can be used to format and display the scores of machine
learning or hypothesis testing experiments.

.. autosummary::
   :toctree: autosummary

   ~skdatasets.utils.scores.scores_table
   ~skdatasets.utils.scores.hypotheses_table
   