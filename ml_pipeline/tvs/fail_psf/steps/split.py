"""
This module defines the following routines used by the 'split' step of the recipe:
"""

from ml_easy.recipes.classification.v1.config import ClassificationSplitConfig
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.split.splitter import DatasetSplitter


def split_fn(conf: ClassificationSplitConfig, context: Context) -> DatasetSplitter:
    return DatasetSplitter(conf.split_ratios[1], conf.split_ratios[2])
