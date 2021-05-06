import dataiku
from dataiku.customrecipe import *

model_input_name = get_input_names_for_role("model")
model_input = [dataiku.Model(name) for name in model_input_name][0]

dataset_input_name = get_input_names_for_role("input_dataset")
dataset_input = [dataiku.Dataset(name) for name in dataset_input_name][0].get_dataframe()

metrics_output = get_output_names_for_role("metrics_dataset")
metrics_output_dataset = [dataiku.Dataset(name) for name in metrics_output][0]

critical_samples = get_output_names_for_role("critical_samples_dataset")
critical_samples_dataset = [dataiku.Dataset(name) for name in critical_samples][0]

metrics_output_dataset.write_with_schema(dataset_input)
critical_samples_dataset.write_with_schema(dataset_input)