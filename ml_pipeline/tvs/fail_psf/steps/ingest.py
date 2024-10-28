"""
This module defines the following routines used by the 'ingest' step of the regression recipe:

"""

from ml_easy.recipes.classification.v1.config import ClassificationIngestConfig
from ml_easy.recipes.interfaces.config import Context
from ml_easy.recipes.steps.ingest.datasets import Dataset, PolarsDataset


def ingest_fn(conf: ClassificationIngestConfig, context: Context) -> Dataset:
    return PolarsDataset.from_sql_database(conf.table_name, conf.credentials.model_dump()).drop_nulls(
        context.target_col
    )
