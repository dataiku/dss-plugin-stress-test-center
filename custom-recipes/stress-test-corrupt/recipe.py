import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_stress_test_center.stress_test_center import FeaturePerturbationTest

recipe_config = get_recipe_config()
test_name = recipe_config.pop("stress_test")
if test_name in FeaturePerturbationTest.TESTS:
    selected_features = recipe_config.pop("selected_features")
    stress_test = FeaturePerturbationTest(
        test_name, params=recipe_config, selected_features=selected_features
    )
else:
    raise ValueError("Unknown stress test %s" % test_name)

input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
corrupted_dataset = dataiku.Dataset(get_output_names_for_role("corrupted_dataset")[0])
corrupted_dataset.write_schema(input_dataset.read_schema())

with corrupted_dataset.get_writer() as writer:
    for df_chunk in input_dataset.iter_dataframes():
        corrupted_dataframe = stress_test.perturb_df(df_chunk)
        writer.write_dataframe(corrupted_dataframe)
