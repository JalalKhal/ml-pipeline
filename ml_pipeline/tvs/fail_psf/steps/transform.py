"""
This module defines the following routines used by the 'transform' step of the recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from typing import Dict, List, Union

from ml_easy.recipes.classification.v1.config import ClassificationTransformConfig
from ml_easy.recipes.constants import FILTER_TO_MODULE
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.transform.filters import EqualFilter, InFilter
from ml_easy.recipes.steps.transform.transformer import (
    FilterTransformer,
    FormaterTransformer,
    MLPipelineTransformer,
    MultipleTfIdfTransformer,
    Transformer,
)
from ml_easy.recipes.utils import load_class


def transformer_fn(conf: ClassificationTransformConfig, context: Context) -> Transformer:
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """

    def load_filters() -> Dict[str, List[Union[EqualFilter[str], InFilter[str]]]]:
        return {
            col: [
                load_class(FILTER_TO_MODULE[f.type])(
                    **{param: value for param, value in f.model_dump().items() if param != 'type'}
                )
                for f in conf.cols[col].filters
            ]
            for col in conf.cols
            if conf.cols[col].filters is not None
        }

    return MLPipelineTransformer(
        [
            (FormaterTransformer(conf), True),
            (FilterTransformer(load_filters()), False),
            (MultipleTfIdfTransformer(conf, context), True),
        ],
        mode=MLPipelineTransformer.Mode.TRAIN,
    )
