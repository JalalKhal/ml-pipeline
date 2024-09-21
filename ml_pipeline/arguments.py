from argparse import ArgumentParser

from ml_pipeline.enums import Profile

parser_main = ArgumentParser()

parser_main.add_argument(
    '--profile',
    type=Profile,
    choices=list(Profile),
    required=True,
    help=('profile of the recipe'),
)
