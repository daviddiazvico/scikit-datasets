"""
Keel datasets (http://sci2s.ugr.es/keel/).

@author: David Diaz Vico
@license: MIT
"""

from .imbalanced import (load_abalone9_18, load_abalone19,
                         load_cleveland_0_vs_4, load_ecoli4,
                         load_ecoli_0_1_3_7_vs_2_6, load_ecoli_0_1_4_6_vs_5,
                         load_ecoli_0_1_4_7_vs_2_3_5_6,
                         load_ecoli_0_1_4_7_vs_5_6, load_ecoli_0_1_vs_5,
                         load_ecoli_0_3_4_6_vs_5, load_ecoli_0_3_4_7_vs_5_6,
                         load_ecoli_0_6_7_vs_5, load_glass2, load_glass4,
                         load_glass5, load_glass_0_1_4_6_vs_2,
                         load_glass_0_1_6_vs_2, load_glass_0_1_6_vs_5,
                         load_glass_0_4_vs_5, load_glass_0_6_vs_5,
                         load_led7digit_0_2_4_5_6_7_8_9_vs_1,
                         load_page_blocks_1_3_vs_4, load_shuttle_c0_vs_c4,
                         load_shuttle_c2_vs_c4, load_vowel0, load_yeast4,
                         load_yeast5, load_yeast6, load_yeast_0_5_6_7_9_vs_4,
                         load_yeast_1_2_8_9_vs_7, load_yeast_1_4_5_8_vs_7,
                         load_yeast_1_vs_7, load_yeast_2_vs_8)
from .standard_classification import (load_balance, load_cleveland, load_ecoli,
                                      load_glass, load_newthyroid,
                                      load_satimage, load_yeast)
