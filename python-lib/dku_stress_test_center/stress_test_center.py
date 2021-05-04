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


class StressTestConfiguration(object):
    def __init__(self,
                 shift_type: Shift,
                 list_of_features: list = None):
        self.shift = shift_type
        self.features = list_of_features  # unused then?
        # check valid configurations


class StressTestGenerator(object):
    def __init__(self,
                 config_list: list,  # list of StressTestConfiguration
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

    def _subsample_clean_df(self,
                            clean_df: pd.DataFrame):

        np.random.seed(self._random_state)

        return clean_df.sample(n=self._clean_dataset_size, random_state=self._random_state)

    def fit_transform(self,
                      clean_df: pd.DataFrame,
                      target_column: str=DkuStressTestCenterConstants.TARGET):

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

        return self.perturbed_datasets_df


def build_stress_metric(y_true: np.array,
                        y_pred: np.array,
                        stress_test_indicator: np.array):
    # batch evaluation
    # return average accuracy drop
    # return average f1 drop
    # return robustness metrics: 1-ASR or imbalanced accuracy for prior shift

    metrics_df = pd.DataFrame(columns=[DkuStressTestCenterConstants.STRESS_TEST_TYPE,
                                       DkuStressTestCenterConstants.ACCURACY_DROP,
                                       DkuStressTestCenterConstants.F1_DROP,
                                       DkuStressTestCenterConstants.ROBUSTNESS])

    clean_filter = stress_test_indicator == DkuStressTestCenterConstants.CLEAN
    clean_y_true = y_true[clean_filter]
    clean_y_pred = y_pred[clean_filter]
    clean_accuracy = accuracy_score(clean_y_true, clean_y_pred)
    clean_f1_score = f1_score(clean_y_true, clean_y_pred)

    stress_ids = stress_test_indicator.unique().drop(DkuStressTestCenterConstants.CLEAN)
    for stress_id in stress_ids:
        stress_filter = stress_test_indicator==stress_id
        stress_y_true = y_true[stress_filter]
        stress_y_pred = y_pred[stress_filter]

        stress_accuracy = accuracy_score(stress_y_true, stress_y_pred)
        stress_f1_score = f1_score(stress_y_true, stress_y_pred)

        stress_acc_drop = stress_accuracy - clean_accuracy
        stress_f1_drop = stress_f1_score - clean_f1_score

        if stress_id == DkuStressTestCenterConstants.PRIOR_SHIFT:
            robustness = balanced_accuracy_score(stress_y_true, stress_y_pred)
        else:
            # 1 - Attack Success Rate
            clean_correctly_predicted_filter = clean_y_pred == clean_y_true
            robustness = accuracy_score(clean_y_pred[clean_correctly_predicted_filter],
                                        stress_y_pred[clean_correctly_predicted_filter])

        metrics_df.append([stress_id, stress_acc_drop, stress_f1_drop, robustness])

    return metrics_df


def get_critical_samples(y_true: np.array,  # needs to be numeric
                         y_pred: np.array,
                         y_proba: np.array,  # n_rows x n_classes
                         stress_test_indicator: np.array,
                         row_indicator: np.array,
                         top_k_samples: int=5):

    # sparse format for the pair (orig_x, pert_x)

    valid_stress_ids = set(stress_test_indicator.unique()) & set(
        DkuStressTestCenterConstants.PERTURBATION_BASED_STRESS_TYPES)
    valid_stress_ids |= DkuStressTestCenterConstants.CLEAN

    true_class_confidence = y_proba[:, y_true]

    true_class_confidence_df = pd.DataFrame(columns=[DkuStressTestCenterConstants.STRESS_TEST_TYPE,
                                                     DkuStressTestCenterConstants.DKU_ROW_ID,
                                                     DkuStressTestCenterConstants.CONFIDENCE],
                                            data=[stress_test_indicator, row_indicator, true_class_confidence])

    true_class_confidence_df = true_class_confidence_df[
        true_class_confidence_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] in valid_stress_ids]

    # critical samples evaluation
    # sort by std of uncertainty

    std_confidence_df = true_class_confidence_df.groupby([DkuStressTestCenterConstants.DKU_ROW_ID]).std()
    uncertainty = std_confidence_df[DkuStressTestCenterConstants.CONFIDENCE]
    grouped_row_indicator = std_confidence_df[DkuStressTestCenterConstants.DKU_ROW_ID]

    critical_samples_df = pd.DataFrame(columns=[DkuStressTestCenterConstants.DKU_ROW_ID,
                                                DkuStressTestCenterConstants.UNCERTAINTY],
                                       data=[grouped_row_indicator, uncertainty])

    critical_samples_df.sort_values(by=DkuStressTestCenterConstants.UNCERTAINTY, ascending=False)

    return critical_samples_df.head(top_k_samples)

