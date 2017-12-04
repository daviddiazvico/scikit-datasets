"""
Keel page-blocks-1-3_vs_4 dataset.

@author: David Diaz Vico
@license: MIT
"""

from ..base import load_imbalanced


def load_page_blocks_1_3_vs_4(return_X_y=False):
    """Load page-blocks-1-3_vs_4 dataset.

    Loads the page-blocks-1-3_vs_4 dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y: arrays
          If return_X_y is True

    """
    return load_imbalanced('page-blocks-1-3_vs_4',
                           'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRhigherThan9p1',
                           names=['Height', 'Lenght', 'Area', 'Eccen',
                                  'P_black', 'P_and', 'Mean_tr', 'Blackpix',
                                  'Blackand', 'Wb_trans', 'Class'],
                           target_names=['Class'], return_X_y=return_X_y)
