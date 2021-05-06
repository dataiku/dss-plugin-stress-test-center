import dataiku
from dataiku.customrecipe import *
from load_parameters import load_samples_column,load_severity,load_attacks_type
from drift_dac.covariate_shift import ReplaceWord,Typos,WordDeletion
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import StressTestGenerator, StressTestConfiguration, build_stress_metric, get_critical_samples
from dku_stress_test_center.utils import DkuStressTestCenterConstants

model_input_name = get_input_names_for_role("model")
model_input = [dataiku.Model(name) for name in model_input_name][0]
target = model_accessor.get_target_variable()

samples_input_name = get_input_names_for_role("input_dataset")
samples_input = [dataiku.Dataset(name) for name in dataset_input_name][0]
samples_df = samples_input.get_dataframe()

samples_column = load_samples_column()
severity = load_severity()

config_list = []
typos_shift,word_swap_shift,word_deletion_shift = load_attacks_type()
if word_swap_shift:
    config_list.append(StressTestConfiguration(ReplaceWord(samples_fraction=severity["word_swap"])))
if typos_shift:
    config_list.append(StressTestConfiguration(Typos(samples_fraction=severity["typos"])))
if word_deletion_shift:
    config_list.append(StressTestConfiguration(WordDeletion(samples_fraction=severity["word_deletion"])))


stressor = StressTestGenerator(config_list=config_list,
                               selected_features = [samples_column],
                               is_categorical = [False],
                               is_text=[True])

perturbed_df = stressor.fit_transform(test_df, target)
perturbed_df_with_prediction = model_accessor.predict(perturbed_df)

# Compute the performance drop metrics
metrics_df = build_stress_metric(y_true=perturbed_df[target],
                                 y_pred=perturbed_df_with_prediction['prediction'],
                                 stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                 pos_label='positive') # TODO this is hardcoding

metrics_output = get_output_names_for_role("metrics_dataset")
metrics_output_dataset = [dataiku.Dataset(name) for name in metrics_output][0]

critical_samples = get_output_names_for_role("critical_samples_dataset")
critical_samples_dataset = [dataiku.Dataset(name) for name in critical_samples][0]

metrics_output_dataset.write_with_schema(metrics_df)
critical_samples_dataset.write_with_schema(dataset_input)