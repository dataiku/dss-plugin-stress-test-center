# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from dku_stress_test_center.utils import DkuStressTestCenterConstants, get_stress_test_name
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from drift_dac.prior_shift import KnockOut
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos


class StressTestConfiguration(object):
    def __init__(self,
                 shift_type: Shift,
                 list_of_features: list = None):
        self.shift = shift_type
        self.features = list_of_features  # unused then?
        # check valid configurations

    @staticmethod
    def create_conf(shift_type, shift_params, shift_features):
        shift = {
            DkuStressTestCenterConstants.ADVERSARIAL: Adversarial,
            DkuStressTestCenterConstants.PRIOR_SHIFT: KnockOut,
            DkuStressTestCenterConstants.MISSING_VALUES: MissingValues,
            DkuStressTestCenterConstants.SCALING: Scaling,
            DkuStressTestCenterConstants.TYPOS: Typos,
            DkuStressTestCenterConstants.REPLACE_WORD: ReplaceWord
        }[shift_type](**shift_params)
        return StressTestConfiguration(shift, shift_features)

class StressTestGenerator(object):
    def __init__(self, clean_dataset_size=DkuStressTestCenterConstants.CLEAN_DATASET_NUM_ROWS,
                random_state=65537):
        self.config_list = None
        self.model_accessor = None
        self.selected_features = None

        self._random_state = random_state
        self._clean_dataset_size = clean_dataset_size

    def set_config(self, shift_configs: dict):
        self.config_list = []
        for shift_type, shift_config in shift_configs.items():
            params, features = shift_config["params"], shift_config.get("selected_features")
            self.config_list.append(StressTestConfiguration.create_conf(shift_type, params, features))

    def _subsample_clean_df(self,
                            clean_df: pd.DataFrame):

        np.random.seed(self._random_state)
        return clean_df.sample(n=self._clean_dataset_size, random_state=self._random_state)

    def fit_transform(self):
        clean_df = self._subsample_clean_df(self.model_accessor.get_original_test_df())
        target_column = self.model_accessor.get_target_variable()

        perturbed_datasets_df = clean_df.copy(deep=True)

        perturbed_datasets_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] = DkuStressTestCenterConstants.CLEAN
        perturbed_datasets_df[DkuStressTestCenterConstants.DKU_ROW_ID] = perturbed_datasets_df.index
        for config in self.config_list:
            pertubed_df = clean_df.copy(deep=True)
            stress_test_id = get_stress_test_name(config.shift)

            if stress_test_id in DkuStressTestCenterConstants.PERTURBATION_BASED_STRESS_TYPES:
                xt = pertubed_df[config.features].values
                yt = pertubed_df[target_column].values
                xt, yt = config.shift.transform(xt, yt)

                pertubed_df.loc[:, config.features] = xt

            else:
                xt = pertubed_df.drop(target_column, axis=1).values
                yt = pertubed_df[target_column].values
                (xt, yt) = config.shift.transform(xt, yt)

                pertubed_df.loc[:, pertubed_df.columns != target_column] = xt

            pertubed_df.loc[:, target_column] = yt

            pertubed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] = stress_test_id
            pertubed_df[DkuStressTestCenterConstants.DKU_ROW_ID] = pertubed_df.index

            # probably need to store the shifted_indices

            perturbed_datasets_df = perturbed_datasets_df.append(pertubed_df)

        return perturbed_datasets_df


def build_stress_metric(y_true: np.array,
                        y_pred: np.array,
                        stress_test_indicator: np.array,
                        pos_label: str or int):
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
    clean_f1_score = f1_score(clean_y_true, clean_y_pred, pos_label=pos_label)

    stress_ids = np.unique(stress_test_indicator)
    stress_ids = np.delete(stress_ids, np.where(stress_ids==DkuStressTestCenterConstants.CLEAN))
    for stress_id in stress_ids:
        stress_filter = stress_test_indicator == stress_id
        stress_y_true = y_true[stress_filter]
        stress_y_pred = y_pred[stress_filter]

        stress_accuracy = accuracy_score(stress_y_true, stress_y_pred)
        stress_f1_score = f1_score(stress_y_true, stress_y_pred, pos_label=pos_label)

        stress_acc_drop = stress_accuracy - clean_accuracy
        stress_f1_drop = stress_f1_score - clean_f1_score

        if stress_id == DkuStressTestCenterConstants.PRIOR_SHIFT:
            robustness = balanced_accuracy_score(stress_y_true, stress_y_pred)
        else:
            # 1 - Attack Success Rate
            clean_correctly_predicted_filter = clean_y_pred == clean_y_true
            robustness = accuracy_score(clean_y_pred[clean_correctly_predicted_filter],
                                        stress_y_pred[clean_correctly_predicted_filter])

        metrics_df = metrics_df.append({
            DkuStressTestCenterConstants.STRESS_TEST_TYPE: stress_id,
            DkuStressTestCenterConstants.ACCURACY_DROP: stress_acc_drop,
            DkuStressTestCenterConstants.F1_DROP: stress_f1_drop,
            DkuStressTestCenterConstants.ROBUSTNESS: robustness
        }, ignore_index=True)

    return metrics_df


def get_critical_samples(y_true_class_confidence: np.array,  # n_rows x 1
                         perturbed_df: pd.DataFrame,
                         top_k_samples: int = 5):
    # sparse format for the pair (orig_x, pert_x)
    stress_test_indicator = perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE]
    valid_stress_ids = set(stress_test_indicator) & set(DkuStressTestCenterConstants.PERTURBATION_BASED_STRESS_TYPES)
    valid_stress_ids |= set([DkuStressTestCenterConstants.CLEAN])

    true_class_confidence_df = pd.DataFrame({
        DkuStressTestCenterConstants.STRESS_TEST_TYPE: stress_test_indicator,
        DkuStressTestCenterConstants.DKU_ROW_ID: perturbed_df[DkuStressTestCenterConstants.DKU_ROW_ID],
        DkuStressTestCenterConstants.CONFIDENCE: y_true_class_confidence
    })

    true_class_confidence_df = true_class_confidence_df[true_class_confidence_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE].isin(valid_stress_ids)]

    # critical samples evaluation sorted by std of uncertainty

    std_confidence_df = true_class_confidence_df.groupby([DkuStressTestCenterConstants.DKU_ROW_ID]).std()
    uncertainty = np.round(std_confidence_df[DkuStressTestCenterConstants.CONFIDENCE].dropna(), 3)

    critical_samples_df = pd.DataFrame({DkuStressTestCenterConstants.UNCERTAINTY: uncertainty})

    if critical_samples_df.empty:
        return pd.DataFrame(), []

    critical_samples_df = critical_samples_df.sort_values(by=DkuStressTestCenterConstants.UNCERTAINTY,
                                                          ascending=False)

    critical_samples_df.reset_index(level=0, inplace=True)

    clean_df_with_id = perturbed_df.loc[perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] == DkuStressTestCenterConstants.CLEAN]
    critical_samples_df = critical_samples_df.merge(clean_df_with_id, on=DkuStressTestCenterConstants.DKU_ROW_ID, how='left').head(top_k_samples)
    return (critical_samples_df.drop([DkuStressTestCenterConstants.DKU_ROW_ID, DkuStressTestCenterConstants.STRESS_TEST_TYPE, DkuStressTestCenterConstants.UNCERTAINTY], axis=1),
        critical_samples_df[DkuStressTestCenterConstants.UNCERTAINTY].tolist())
