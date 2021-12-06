"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools as it
import sys
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (friedmanchisquare, kruskal, mannwhitneyu, rankdata,
                         wilcoxon)
from statsmodels.sandbox.stats.multicomp import multipletests

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


def scores_table(
    datasets: Sequence[str],
    estimators: Sequence[str],
    scores: np.typing.ArrayLike,
    stds: Optional[np.typing.ArrayLike] = None,
    greater_is_better: bool = True,
    method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'average',
    score_decimals: int = 2,
    rank_decimals: int = 0,
) -> pd.DataFrame:
    """
    Scores table.

    Prints a table where each row represents a dataset and each column
    represents an estimator.

    Parameters
    ----------
    datasets: array-like
        List of dataset names.
    estimators: array-like
        List of estimator names.
    scores: array-like
        Matrix of scores where each column represents a model.
    stds: array_like, default=None
        Matrix of standard deviations where each column represents a
        model.
    greater_is_better: boolean, default=True
        Whether a greater score is better (score) or worse
        (loss).
    method: {'average', 'min', 'max', 'dense', 'ordinal'}, default='average'
        Method used to solve ties.

    Returns
    -------
    table: array-like
        Table of mean and standard deviation of each estimator-dataset
        pair. A ranking of estimators is also generated.

    """
    scores = np.asanyarray(scores)
    stds = None if stds is None else np.asanyarray(stds)

    ranks = np.asarray([
        rankdata(-m, method=method)
        if greater_is_better
        else rankdata(m, method=method)
        for m in scores
    ])
    table = pd.DataFrame(data=scores, index=datasets, columns=estimators)
    for i, d in enumerate(datasets):
        for j, e in enumerate(estimators):
            table.loc[d, e] = f'{scores[i, j]:.{score_decimals}f}'
            if stds is not None:
                table.loc[d, e] += f' ±{stds[i, j]:.{score_decimals}f}'
            table.loc[d, e] += f' ({ranks[i, j]:.{rank_decimals}f})'
    table.loc['rank mean'] = np.around(
        np.mean(ranks, axis=0),
        decimals=score_decimals,
    )
    return table


def hypotheses_table(
    samples: np.typing.ArrayLike,
    models: Sequence[str],
    alpha: float = 0.05,
    multitest: Literal[None, 'kruskal', 'friedmanchisquare'] = None,
    test: Literal['mannwhitneyu', 'wilcoxon'] = 'wilcoxon',
    correction: Literal[
        None, 'bonferroni', 'sidak', 'holm-sidak', 'holm',
        'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh',
        'fdr_tsbky',
    ] = None,
    multitest_args: Mapping[str, Any] = dict(),
    test_args: Mapping[str, Any] = dict(),
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """ 
    Hypotheses table.

    Prints a hypothesis table with a selected test and correction.

    Parameters
    ----------
    samples: array-like
        Matrix of samples where each column represent a model.
    models: array-like
        Model names.
    alpha: float in [0, 1], default=0.05
        Significance level.
    multitest: {'kruskal', 'friedmanchisquare'}, default=None
        Ranking multitest used.
    test: {'mannwhitneyu', 'wilcoxon'}, default='wilcoxon'
        Ranking test used.
    correction: {'bonferroni', 'sidak', 'holm-sidak', 'holm',
                 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh',
                 'fdr_tsbky'}, default=None
        Method used to adjust the p-values.
    multitest_args: dict
        Optional ranking test arguments.
    test_args: dict
        Optional ranking test arguments.

    Returns
    -------
    multitest_table: array-like
        Table of p-value and rejection/non-rejection for the
        multitest hypothesis.
    test_table: array-like
        Table of p-values and rejection/non-rejection for each test
        hypothesis.
    """
    samples = np.asanyarray(samples)

    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [models[vs[0]] + " vs " + models[vs[1]] for vs in versus]
    multitests = {'kruskal': kruskal, 'friedmanchisquare': friedmanchisquare}
    tests = {'mannwhitneyu': mannwhitneyu, 'wilcoxon': wilcoxon}
    multitest_table = None
    if multitest is not None:
        multitest_table = pd.DataFrame(
            index=[multitest],
            columns=['p-value', 'Hypothesis'],
        )
        statistic, pvalue = multitests[multitest](*samples, **multitest_args)
        reject_str = 'Rejected' if pvalue <= alpha else 'Not rejected'
        multitest_table.loc[multitest] = ['{0:.2f}'.format(pvalue), reject_str]
        if pvalue > alpha:
            return multitest_table, None
    pvalues = [tests[test](
        samples[:, vs[0]],
        samples[:, vs[1]],
        **test_args,
    )[1] for vs in versus]
    if correction is not None:
        reject, pvalues, _, _ = multipletests(
            pvalues,
            alpha,
            method=correction,
        )
    else:
        reject = [
            'Rejected'
            if pvalue <= alpha
            else 'Not rejected'
            for pvalue in pvalues
        ]
    test_table = pd.DataFrame(
        index=comparisons,
        columns=['p-value', 'Hypothesis'],
    )
    for i, d in enumerate(comparisons):
        test_table.loc[d] = ['{0:.2f}'.format(pvalues[i]), reject[i]]
    return multitest_table, test_table
