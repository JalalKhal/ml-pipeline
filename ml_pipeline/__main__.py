from ml_easy.recipes.interfaces.recipe import RecipeFactory
from ml_easy.recipes.steps.steps_config import RecipePathsConfig

from ml_pipeline.arguments import parser_main


def main():
    args = parser_main.parse_args()
    paths_config = RecipePathsConfig(recipe_root_path='./ml_pipeline/tvs/fail_psf', profile=args.profile)
    recipe = RecipeFactory.create_recipe(paths_config)
    recipe.run()


if __name__ == '__main__':
    main()
