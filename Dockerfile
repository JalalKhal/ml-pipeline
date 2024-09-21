FROM python:3.12.3
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install poetry==1.8.3

WORKDIR /home/ml-pipeline

COPY poetry.lock pyproject.toml  ./
COPY ml_pipeline ./ml_pipeline
RUN touch README.md

RUN poetry install --without test_dep,format_and_typing_dep
