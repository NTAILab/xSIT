from typing import Literal
import numpy as np


PREPROCESSING_FUNCTIONS = {
    None: lambda xs: xs,
    'concat_negative': lambda xs: np.concatenate([xs, -xs], axis=1),
}


FeaturePreprocessingT = Literal['concat_negative'] | None


def make_preprocessing_ft(ft: FeaturePreprocessingT):
    if callable(ft):
        return ft
    return PREPROCESSING_FUNCTIONS[ft]
