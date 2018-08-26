"""
Gunnar Raetsch benchmark datasets
(https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets).

@author: David Diaz Vico
@license: MIT
"""

from functools import partial

from .base import load_dataset


load_banana = partial(load_dataset, name='banana')
load_breast_cancer = partial(load_dataset, name='breast_cancer')
load_diabetis = partial(load_dataset, name='diabetis')
load_flare_solar = partial(load_dataset, name='flare_solar')
load_german = partial(load_dataset, name='german')
load_heart = partial(load_dataset, name='heart')
load_image = partial(load_dataset, name='image')
load_ringnorm = partial(load_dataset, name='ringnorm')
load_splice = partial(load_dataset, name='splice')
load_thyroid = partial(load_dataset, name='thyroid')
load_titanic = partial(load_dataset, name='titanic')
load_twonorm = partial(load_dataset, name='twonorm')
load_waveform = partial(load_dataset, name='waveform')


load = {'banana': load_banana, 'breast_cancer': load_breast_cancer,
        'diabetis': load_diabetis, 'flare_solar': load_flare_solar,
        'german': load_german, 'heart': load_heart, 'image': load_image,
        'ringnorm': load_ringnorm, 'splice': load_splice,
        'thyroid': load_thyroid, 'titanic': load_titanic,
        'twonorm': load_twonorm, 'waveform': load_waveform}
