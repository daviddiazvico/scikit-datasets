"""
@author: David Diaz Vico
@license: MIT
"""
from __future__ import annotations

import itertools as it
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, overload

import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    rankdata,
    wilcoxon,
)
from scipy.stats.stats import ttest_ind_from_stats, ttest_rel
from statsmodels.sandbox.stats.multicomp import multipletests

CorrectionLike = Literal[
    None,
    'bonferroni',
    'sidak',
    'holm-sidak',
    'holm',
    'simes-hochberg',
    'hommel',
    'fdr_bh',
    'fdr_by',
    'fdr_tsbh',
    'fdr_tsbky',
]

MultitestLike = Literal['kruskal', 'friedmanchisquare']

TestLike = Literal['mannwhitneyu', 'wilcoxon']


def _stylefun(
    table: pd.DataFrame,
    *,
    style: str,
    mask: np.typing.NDArray[bool],
) -> np.typing.NDArray[str]:
    full_mask = np.zeros(table.shape, dtype=np.bool_)
    full_mask[:mask.shape[0]] = mask
    style_table = np.full(shape=table.shape, fill_value=style)
    style_table[~full_mask] = ""

    return style_table


def scores_table(
    *,
    datasets: Sequence[str],
    estimators: Sequence[str],
    scores: np.typing.ArrayLike,
    stds: np.typing.ArrayLike | None = None,
    nobs: int | None = None,
    greater_is_better: bool = True,
    method: Literal['average', 'min', 'max', 'dense', 'ordinal'] = 'min',
    first_style: str | None = None,
    second_style: str | None = None,
    mark_significant: bool = False,
    paired_test: bool = False,
    significancy_level: float = 0.01,
    score_decimals: int = 2,
    rank_decimals: int = 0,
    add_score_mean: bool = False,
    two_sided: bool = True,
    return_styler: bool | Literal["auto"] = "auto",
) -> pd.DataFrame | pd.io.formats.style.Styler:
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

    if return_styler == "auto":
        return_styler = first_style is not None

    assert scores.ndim in {2, 3}
    score_means = scores if scores.ndim == 2 else np.mean(scores, axis=-1)
    if scores.ndim == 3:
        assert stds is None
        assert nobs is None
        stds = np.std(scores, axis=-1)
        nobs = scores.shape[-1]

    ranks = np.asarray([
        rankdata(-m, method=method)
        if greater_is_better
        else rankdata(m, method=method)
        for m in score_means
    ])

    table = pd.DataFrame(data=score_means, index=datasets, columns=estimators)
    for i, d in enumerate(datasets):
        for j, e in enumerate(estimators):
            table.loc[d, e] = f'{{{score_means[i, j]:.{score_decimals}f}'
            if stds is not None:
                table.loc[d, e] += f' ± {stds[i, j]:.{score_decimals}f}'
            table.loc[d, e] += f'}} ({ranks[i, j]:.{rank_decimals}f})'

    score_mean = np.mean(score_means, axis=0)
    rank_mean = np.mean(ranks, axis=0)

    score_mean_rank = (
        rankdata(-score_mean, method=method)
        if greater_is_better
        else rankdata(score_mean, method=method)
    )

    rank_mean_rank = rankdata(rank_mean, method=method)

    if add_score_mean:
        table.loc["Average accuracy"] = [
            f"{{{float(m):.{score_decimals}f}}}" for m in score_mean
        ]

    table.loc["Average rank"] = [
        f"{{{float(m):.{score_decimals}f}}}" for m in rank_mean
    ]

    n_extra_rows = 2 if add_score_mean else 1

    last_rows_mask = np.zeros(table.shape, dtype=bool)
    last_rows_mask[-n_extra_rows:, :] = 1

    if mark_significant:

        firsts = (ranks == 1)

        for i, (mean_row, std_row, rank_row, score_row) in enumerate(
            zip(score_means, stds, ranks, scores),
        ):
            # Break ties by greater std
            sorted_ranks = sorted(
                range(len(rank_row)),
                key=lambda ind: (rank_row[ind], -std_row[ind]),
            )

            first_index = sorted_ranks[0]
            second_index = sorted_ranks[1]

            alternative = "two-sided" if two_sided else "greater"

            if paired_test:
                assert score_row.ndim == 2

                scores1 = score_row[first_index]
                scores2 = score_row[second_index]

                _, pvalue = ttest_rel(
                    scores1,
                    scores2,
                    axis=-1,
                    alternative=alternative,
                )

            else:
                mean1 = mean_row[first_index]
                mean2 = mean_row[second_index]
                std1 = std_row[first_index]
                std2 = std_row[second_index]

                assert nobs
                _, pvalue = ttest_ind_from_stats(
                    mean1=mean1,
                    std1=std1,
                    nobs1=nobs,
                    mean2=mean2,
                    std2=std2,
                    nobs2=nobs,
                    equal_var=False,
                    alternative=alternative,
                )

            if pvalue < significancy_level:
                # Significant
                first_index = np.nonzero(firsts[i])[0][0]

                table.iloc[
                    i,
                    first_index,
                ] = f"\\hphantom{{*}}{table.iloc[i, first_index]}*"

    styler = table.style.apply(
        _stylefun,
        axis=None,
        style="itshape:",
        mask=last_rows_mask,
    ).apply_index(
        _stylefun,
        axis=0,
        style="itshape:",
        mask=last_rows_mask[:, 0],
    ).apply_index(
        _stylefun,
        axis=0,
        style="bfseries:",
        mask=last_rows_mask[:, 0],
    ).apply_index(
        lambda x: ["bfseries:"] * len(x),
        axis=1,
    )

    if first_style:
        firsts = np.vstack(
            [ranks == 1, score_mean_rank == 1, rank_mean_rank == 1],
        )

        styler = styler.apply(
            _stylefun,
            axis=None,
            style=first_style,
            mask=firsts,
        )

    if second_style:
        seconds = np.vstack(
            [ranks == 2, score_mean_rank == 2, rank_mean_rank == 2],
        )

        styler = styler.apply(
            _stylefun,
            axis=None,
            style=second_style,
            mask=seconds,
        )

    return styler if return_styler else table


def hypotheses_table(
    samples: np.typing.ArrayLike,
    models: Sequence[str],
    *,
    alpha: float = 0.05,
    multitest: Optional[MultitestLike] = None,
    test: TestLike = 'wilcoxon',
    correction: CorrectionLike = None,
    multitest_args: Optional[Mapping[str, Any]] = None,
    test_args: Optional[Mapping[str, Any]] = None,
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
    correction: {'bonferroni', 'sidak', 'holm-sidak', 'holm', \
                 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', \
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
    if multitest_args is None:
        multitest_args = {}

    if test_args is None:
        test_args = {}

    samples = np.asanyarray(samples)

    versus = list(it.combinations(range(len(models)), 2))
    comparisons = [
        f"{models[first]} vs {models[second]}"
        for first, second in versus
    ]

    multitests = {
        'kruskal': kruskal,
        'friedmanchisquare': friedmanchisquare,
    }
    tests = {
        'mannwhitneyu': mannwhitneyu,
        'wilcoxon': wilcoxon,
    }

    multitest_table = None
    if multitest is not None:
        multitest_table = pd.DataFrame(
            index=[multitest],
            columns=['p-value', 'Hypothesis'],
        )
        _, pvalue = multitests[multitest](
            *samples.T,
            **multitest_args,
        )
        reject_str = 'Rejected' if pvalue <= alpha else 'Not rejected'
        multitest_table.loc[multitest] = ['{0:.2f}'.format(pvalue), reject_str]

        # If the multitest does not detect a significative difference,
        # the individual tests are not meaningful, so skip them.
        if pvalue > alpha:
            return multitest_table, None

    pvalues = [
        tests[test](
            samples[:, first],
            samples[:, second],
            **test_args,
        )[1] for first, second in versus
    ]

    if correction is not None:
        reject_bool, pvalues, _, _ = multipletests(
            pvalues,
            alpha,
            method=correction,
        )
        reject = [
            'Rejected'
            if r
            else 'Not rejected'
            for r in reject_bool
        ]
    else:
        reject = [
            'Rejected'
            if pvalue <= alpha
            else 'Not rejected'
            for pvalue in pvalues
        ]

    data = [
        ('{0:.2f}'.format(p), r)
        for p, r in zip(pvalues, reject)
    ]

    test_table = pd.DataFrame(
        data,
        index=comparisons,
        columns=['p-value', 'Hypothesis'],
    )

    return multitest_table, test_table
