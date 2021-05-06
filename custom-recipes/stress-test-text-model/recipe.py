import dataiku
from dataiku.customrecipe import *
from dku_config_parameters.load_parameters import DkuConfigLoadingStressTestCenter
from drift_dac.covariate_shift import ReplaceWord, Typos, WordDeletion
from dku_stress_test_center.stress_test_center import (
    StressTestConfiguration,
    StressTestGenerator,
    build_stress_metric,
    get_critical_samples,
)
from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from model_metadata import get_model_handler
import numpy as np

# load recipe datasets and parameters
dku_config = DkuConfigLoadingStressTestCenter()
settings = dku_config.load_settings()

# get model target
model_handler = get_model_handler(model=settings.input_model, version_id=None)
model_accessor = ModelAccessor(model_handler)
target = model_accessor.get_target_variable()

# instanciate the class implementing the perturbations to apply
config_list = []
if settings.word_swap:
    config_list.append(
        StressTestConfiguration(
            ReplaceWord(samples_fraction=settings.word_swap_severity)
        )
    )
if settings.typos:
    config_list.append(
        StressTestConfiguration(Typos(samples_fraction=settings.typos_severity))
    )
if settings.word_deletion:
    config_list.append(
        StressTestConfiguration(
            WordDeletion(samples_fraction=settings.word_deletion_severity)
        )
    )


stressor = StressTestGenerator(
    config_list=config_list,
    selected_features=[settings.samples_column],
    is_categorical=np.array([False]),
    is_text=np.array([True]),
)

# apply perturbation to the input dataset
samples_df = settings.input_dataset.get_dataframe()
perturbed_df = stressor.fit_transform(samples_df, target)
perturbed_df_with_prediction = model_accessor.predict(perturbed_df)

# compute metrics
metrics_df = build_stress_metric(
    y_true=perturbed_df[target],
    y_pred=perturbed_df_with_prediction["prediction"],
    stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
    pos_label="positive",
)  # TODO this is hardcoding

# compute critical samples
name_mapping = {
    "REPLACE_WORD": "Adversarial attack",
    "TYPOS": "Missing values",
    "WORD_DELETION": "Word deletion",
}

metrics_list = []
for index, row in metrics_df.iterrows():
    dct = {}
    dct["attack_type"] = name_mapping.get(row["_dku_stress_test_type"])
    dct["accuracy_drop"] = 100 * round(row["accuracy_drop"], 3)
    dct["robustness"] = 100 * round(row["robustness"], 3)
    metrics_list.append(dct)

y_true = perturbed_df[target]
y_true_class_confidence = perturbed_df_with_prediction[["proba_0", "proba_1"]].values
y_true_idx = np.array([[True, False] if y == "1" else [False, True] for y in y_true])
y_true_class_confidence = y_true_class_confidence[y_true_idx]

critical_samples_id_df = get_critical_samples(
    y_true_class_confidence=y_true_class_confidence,
    stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
    row_indicator=perturbed_df[DkuStressTestCenterConstants.DKU_ROW_ID],
)

# TODO hot fix that should be done inside get_critical_samples
critical_samples_id_df.reset_index(level=0, inplace=True)

clean_df_with_id = perturbed_df.loc[
    perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE]
    == DkuStressTestCenterConstants.CLEAN
].drop(DkuStressTestCenterConstants.STRESS_TEST_TYPE, axis=1)
critical_samples_df = critical_samples_id_df.merge(
    clean_df_with_id, on=DkuStressTestCenterConstants.DKU_ROW_ID, how="left"
).drop(DkuStressTestCenterConstants.DKU_ROW_ID, axis=1)

# write output datasets
settings.metrics_output_dataset.write_with_schema(metrics_df)
settings.critical_samples_dataset.write_with_schema(critical_samples_df)