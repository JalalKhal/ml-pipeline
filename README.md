# ML Pipeline Project

This repository contains a machine learning pipeline project based on an MLflow Recipes template and the ml-easy library. 
The project is structured for classification tasks and includes multiple components such as data ingestion, transformation, model training, evaluation, and registration.


## Running the Pipeline

The pipeline can be run by executing the main entry point located in `ml_pipeline/__main__.py`. This script orchestrates the flow of the pipeline, including argument parsing, loading configurations, and executing the various steps (ingestion, transformation, model training, etc.).



### Running the Pipeline

You can run the pipeline with:

```bash
poetry run python -m ml_pipeline --profile <profile_name>
```

Replace `<profile_name>` with the appropriate profile defined in the `profiles/` directory. Example profiles include `local.yaml`.

### Docker Execution

To run the pipeline within a Docker container, you can build and run the container using the provided `Dockerfile`:


## Configuration

The pipeline is configured using a set of YAML files:

- **recipe.yaml**: Main configuration file that defines the steps of the pipeline, credentials for database access, and feature transformations.
- **profiles/local.yaml**: Example profile that defines environment-specific variables like experiment tracking URIs, and more.


## Pipeline Steps

The following steps are implemented in the pipeline:

1. **Ingest**: Loads data. See `ingest.py`.
2. **Transform**: Applies feature transformation (e.g., text vectorization). See `transform.py`.
3. **Split**: Splits the dataset into training, validation, and test sets. See `split.py`.
4. **Train**: Trains a classification model (e.g., logistic regression). See `train.py`.
5. **Evaluate**: Evaluates the model using metrics like accuracy and F1 score.
6. **Register**: Registers the trained model in an MLflow model registry. See `register_.py`.

## Example YAML Configuration (`recipe.yaml`)

The main configuration file `recipe.yaml` defines the steps and parameters for the pipeline:

```yaml
recipe: "classification/v1"

context:
  recipe_root_path : "./ml_pipeline/tvs/fail_psf"
  target_col: {{TARGET_COL}}
  experiment: {{EXPERIMENT}}

steps:
  ingest:
    ingest_fn: "ingest_fn"
    table_name: "dataset"
    credentials:
      username: "{{ 'DB_USERNAME' | env }}"
      password: "{{ 'DB_PASSWORD' | env }}"
  split:
    split_fn: "split_fn"
    split_ratios: [ 0.75, 0.125, 0.125 ]
  transform:
    transformer_fn: "transformer_fn"
    cols:
      cm_motif_dep:
        embedder:
          path: "sklearn.feature_extraction.text.TfidfVectorizer"
          params:
            max_features: 5000
            ngram_range: (1,3)
  train:
    estimator_fn: "estimator_fn"
    loss: "log_loss"
    validation_metric: &validation_metric
      name: "accuracy_score"
  evaluate:
    validation_criteria:
      - metric: *validation_metric
        threshold: .5
```



