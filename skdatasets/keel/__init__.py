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


load = {'abalone9-18': load_abalone9_18, 'abalone19': load_abalone19,
        'cleveland-0_vs_4': load_cleveland_0_vs_4, 'ecoli4': load_ecoli4,
        'ecoli-0-1-3-7_vs_2-6': load_ecoli_0_1_3_7_vs_2_6,
        'ecoli-0-1-4-6_vs_5': load_ecoli_0_1_4_6_vs_5,
        'ecoli-0-1-4-7_vs_2-3-5-6': load_ecoli_0_1_4_7_vs_2_3_5_6,
        'ecoli-0-1-4-7_vs_5-6': load_ecoli_0_1_4_7_vs_5_6,
        'ecoli-0-1_vs_5': load_ecoli_0_1_vs_5,
        'ecoli-0-3-4-6_vs_5': load_ecoli_0_3_4_6_vs_5,
        'ecoli-0-3-4-7_vs_5-6': load_ecoli_0_3_4_7_vs_5_6,
        'ecoli-0-6-7_vs_5': load_ecoli_0_6_7_vs_5, 'glass2': load_glass2,
        'glass4': load_glass4, 'glass5': load_glass5,
        'glass-0-1-4-6_vs_2': load_glass_0_1_4_6_vs_2,
        'glass-0-1-6_vs_2': load_glass_0_1_6_vs_2,
        'glass-0-1-6_vs_5': load_glass_0_1_6_vs_5,
        'glass-0-4_vs_5': load_glass_0_4_vs_5,
        'glass-0-6_vs_5': load_glass_0_6_vs_5,
        'led7digit-0-2-4-5-6-7-8-9_vs_1': load_led7digit_0_2_4_5_6_7_8_9_vs_1,
        'page-blocks-1-3_vs_4': load_page_blocks_1_3_vs_4,
        'shuttle-c0-vs-c4': load_shuttle_c0_vs_c4,
        'shuttle-c2-vs-c4': load_shuttle_c2_vs_c4, 'vowel0': load_vowel0,
        'yeast4': load_yeast4, 'yeast5': load_yeast5, 'yeast6': load_yeast6,
        'yeast-0-5-6-7-9_vs_4': load_yeast_0_5_6_7_9_vs_4,
        'yeast-1-2-8-9_vs_7': load_yeast_1_2_8_9_vs_7,
        'yeast-1-4-5-8_vs_7': load_yeast_1_4_5_8_vs_7,
        'yeast-1_vs_7': load_yeast_1_vs_7, 'yeast-2_vs_8': load_yeast_2_vs_8,
        'balance': load_balance, 'cleveland': load_cleveland,
        'ecoli': load_ecoli, 'glass': load_glass, 'newthyroid': load_newthyroid,
        'satimage': load_satimage, 'yeast': load_yeast}
