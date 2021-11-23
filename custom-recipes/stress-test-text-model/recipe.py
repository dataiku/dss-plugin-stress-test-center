import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
from dku_stress_test_center.stress_test_center import FeaturePerturbationTest
from dku_stress_test_center.utils import DkuStressTestCenterConstants

input_dataset = dataiku.Dataset(get_input_names_for_role("input_dataset")[0])
input_dataframe = input_dataset.get_dataframe()

corrupted_dataset = dataiku.Dataset(get_output_names_for_role("corrupted_dataset")[0])
recipe_config = get_recipe_config()

test_class, _ = DkuStressTestCenterConstants.TESTS[recipe_config["stress_test"]]
feature_perturbation = test_class(samples_fraction=recipe_config["samples_fraction"])
stress_test = FeaturePerturbationTest(feature_perturbation, recipe_config["selected_features"])
corrupted_dataframe = stress_test.perturb_df(input_dataframe)

corrupted_dataset.write_with_schema(corrupted_dataframe) # TODO: by batches?
