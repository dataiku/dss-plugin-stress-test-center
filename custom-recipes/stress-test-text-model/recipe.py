import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_stress_test_center.stress_test_center import FeaturePerturbationTest
from dku_stress_test_center.utils import DkuStressTestCenterConstants

recipe_config = get_recipe_config()
test_class, _ = DkuStressTestCenterConstants.TESTS[recipe_config["stress_test"]]
feature_perturbation = test_class(samples_fraction=recipe_config["samples_fraction"])
stress_test = FeaturePerturbationTest(feature_perturbation, recipe_config["selected_features"])

input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
corrupted_dataset = dataiku.Dataset(get_output_names_for_role("corrupted_dataset")[0])
corrupted_dataset.write_schema(input_dataset.read_schema())

with corrupted_dataset.get_writer() as writer:
    for df_chunk in input_dataset.iter_dataframes():
        corrupted_dataframe = stress_test.perturb_df(df_chunk)
        writer.write_dataframe(corrupted_dataframe) # TODO: by batches?
