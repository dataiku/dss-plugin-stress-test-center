# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import copy
from dku_stress_test_center.utils import DkuStressTestCenterConstants, safe_str, get_stress_test_name
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Stress Test Center Plugin | %(levelname)s - %(message)s')


'''python

# Get test data
test_df =  model_accessor.get_original_test_df()
selected_features = ... # either top 10 feats or later given by the user.
test_df = test_df[selected_features] 
​
​
# Run the stress tests
config_list = [StressTestConfiguration(PriorShift), StressTestConfiguration(MissingValues)]
stressor = StressTestGenerator(config_list)
perturbed_df = stressor.fit_transform(test_df) # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_
​
perturbed_df_with_prediction = model_accessor.predict(perturbed_df)
​
# Compute the performance drop metrics
metrics_df = build_stress_metric(y_pred = perturbed_df_with_prediction['target'], 
								stress_test_indicator = perturbed_df_with_prediction['_STRESS_TEST_TYPE'], 
								row_indicator = perturbed_df_with_prediction['_DKU_ROW_ID_'])
​
critical_samples = get_critical_samples(y_pred = perturbed_df_with_prediction['target'], 
										stress_test_indicator = perturbed_df_with_prediction['_STRESS_TEST_TYPE'], 
										row_indicator = perturbed_df_with_prediction['_DKU_ROW_ID_']) # sparse format for the pair (orig_x, pert_x) ?


'''


class StressTestConfiguration(object):
    def __init__(self,
                 shift_type: Shift,
                 list_of_features: list=None):
        self.shift = shift_type
        self.features = list_of_features # unused then?
        # check valid configurations


class StressTestGenerator(object):
    def __init__(self,
                 config_list: list, # list of StressTestConfiguration
                 is_categorical: np.array,
                 is_text: np.array,
                 clean_dataset_size=DkuStressTestCenterConstants.CLEAN_DATASET_NUM_ROWS,
                 random_state=65537):

        self.config_list = config_list
        self.is_categorical = is_categorical
        self.is_text = is_text
        self.is_numeric = ~is_categorical and ~is_text
        self.perturbed_datasets_df = None

        self._random_state = random_state
        self._clean_dataset_size = clean_dataset_size

    def _subsample_clean_df(self, clean_df):

        np.random.seed(self._random_state)

        return clean_df.sample(n=self._clean_dataset_size, random_state=self._random_state)

    def fit_transform(self, clean_df, target_column='target'):

        clean_df = self._subsample_clean_df(clean_df)

        self.perturbed_datasets_df = clean_df.copy()
        self.perturbed_datasets_df = self.perturbed_datasets_df.reset_index(True)

        self.perturbed_datasets_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] = DkuStressTestCenterConstants.CLEAN
        self.perturbed_datasets_df[DkuStressTestCenterConstants.DKU_ROW_ID] = self.perturbed_datasets_df.index

        clean_x = clean_df.drop(target_column, axis=1).values
        clean_y = clean_df[target_column].values

        for config in self.config_list:
            xt = copy.deepcopy(clean_x)
            yt = copy.deepcopy(clean_y)

            if config.shift.feature_type == PerturbationConstants.NUMERIC:
                (xt[:, self.is_numeric], yt) = config.shift.transform(xt[:, self.is_numeric].astype(float), yt)
            elif config.shift.feature_type == PerturbationConstants.CATEGORICAL:
                (xt[:, self.is_categorical], yt) = config.shift.transform(xt[:, self.is_categorical], yt)
            #elif config.shift.feature_type == PerturbationConstants.TEXT:
            #    (xt[:, self.is_text], yt) = config.shift.transform(xt[:, self.is_text], yt)
            else:
                (xt[:, self.is_text], yt) = config.shift.transform(xt[:, self.is_text], yt)
                xt[:, config.features], yt = config.shift.transform(xt[:, config.features], yt)

            pertubed_df = pd.DataFrame(columns=clean_df.columns)
            pertubed_df[[clean_x.columns]].values = xt
            pertubed_df[[target_column]].values = yt

            [DkuStressTestCenterConstants.STRESS_TEST_TYPE] = get_stress_test_name(config.shift)
            pertubed_df[DkuStressTestCenterConstants.DKU_ROW_ID] = pertubed_df.index

            # probably need to store the shifted_indices

            self.perturbed_datasets_df.append(pertubed_df)


def build_stress_metric(y_true,
                        y_pred,
                        list_y_proba,
						stress_test_indicator,
						row_indicator):
    # batch evaluation
    # return average accuracy drop
    # return average f1 drop
    # return robustness metrics: 1-ASR or imbalanced accuracy for prior shift
    self.metrics_clean['accuracy'] = accuracy_score(clean_y, clean_y_pred)
    self.metrics_clean['f1_score'] = f1_score(clean_y, clean_y_pred)
    self.metrics_clean['balanced_accuracy'] = balanced_accuracy_score(clean_y, clean_y_pred)

    return metrics_df
​

def get_critical_samples(y_true,
                        y_pred,
                        list_y_proba,
						stress_test_indicator,
						row_indicator):

    # sparse format for the pair (orig_x, pert_x)

    if config.shift.shifted_indices is not None:  # it's perturbation based

        # critical samples evaluation
        # sort by std of uncertainty
        # store proba per sample
        self.sample_predictions = clean_y_pred
        self.sample_true_class_confidence = clean_y_proba[:, [model.classes_.index(true_y) for true_y in clean_y]]

    return critical_samples






