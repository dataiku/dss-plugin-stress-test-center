# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_stress_test_center.metrics import worst_group_performance, performance_variation, stress_resilience
from drift_dac.perturbation_shared_utils import Shift
from dku_webapp import DkuWebappConstants

class StressTest(object):
    def __init__(self, shift: Shift, list_of_features: list = None):
        self.shift = shift
        self.features = list_of_features
        self.df_with_pred = None
        self.sample_perturbation = shift.__class__ in DkuStressTestCenterConstants.FEATURE_PERTURBATIONS
        # TODO: check valid configurations

    @staticmethod
    def create(shift_type, shift_params, shift_features):
        shift = DkuStressTestCenterConstants.TESTS[shift_type](**shift_params)
        return StressTest(shift, shift_features)

    def perturb(self, df: pd.DataFrame, target: str):
        if self.sample_perturbation:
            perturbed_columns = self.features
        else:
            perturbed_columns = df.columns != target

        xt = df.loc[:, perturbed_columns].values
        yt = df[target].values
        xt, yt = self.shift.transform(xt, yt)

        df.loc[:, perturbed_columns] = xt
        df.loc[:, target] = yt

        return df

    def compute_metrics(self, clean_df: pd.DataFrame, target: str, perf_metric: str):
        clean_y_true = clean_df[target]
        clean_y_pred = clean_df[DkuWebappConstants.PREDICTION]
        perturbed_y_true = self.df_with_pred[target]
        perturbed_y_pred = self.df_with_pred[DkuWebappConstants.PREDICTION]
        return {
            "robustness": self.compute_robustness(
                perf_metric, clean_y_true, clean_y_pred, perturbed_y_true, perturbed_y_pred
            ),
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, clean_y_pred, perturbed_y_true, perturbed_y_pred
            )
        }

    def compute_robustness(self, perf_metric: str, clean_y_true: np.array, clean_y_pred: np.array,
                           perturbed_y_true: np.array, perturbed_y_pred: np.array):
        if self.sample_perturbation:
            return stress_resilience(clean_y_pred, perturbed_y_pred)
        return worst_group_performance(perf_metric, perturbed_y_true, perturbed_y_pred)


class StressTestGenerator(object):
    def __init__(self, random_state=65537):
        self._random_state = random_state

        self.tests = None
        self.model_accessor = None
        self._sampling_proportion = None
        self._clean_df = None

    def set_config(self, config: dict):
        self._sampling_proportion = config["samples"]
        tests_config = config["perturbations"]
        self.tests = []
        for shift_type, shift_config in tests_config.items():
            params, features = shift_config["params"], shift_config.get("selected_features")
            self.tests.append(StressTest.create(shift_type, params, features))

    def compute_test_metrics(self, test: StressTest):
        target = self.model_accessor.get_target_variable()
        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
        perturbed_df = test.perturb(df, target)
        test.df_with_pred = self.model_accessor.predict_and_concatenate(perturbed_df).dropna()
        clean_df_with_pred = self._clean_df.loc[test.df_with_pred.index, :]

        return {
            test.shift.__class__.__name__: test.compute_metrics(clean_df_with_pred, target, "auc") # TODO: use get_evaluation_metric
        }

    def predict_clean_df(self):
        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
        self._clean_df = self.model_accessor.predict_and_concatenate(df).dropna()

    def build_stress_metrics(self):
        metrics = {
            DkuWebappConstants.SAMPLE_PERTURBATION: {"metrics": {}},
            DkuWebappConstants.SUBPOPULATION_PERTURBATION: {"metrics": {}}
        }

        self.predict_clean_df()

        for test in self.tests:
            test_type = DkuWebappConstants.SAMPLE_PERTURBATION if test.sample_perturbation else DkuWebappConstants.SUBPOPULATION_PERTURBATION
            metrics[test_type]["metrics"] = self.compute_test_metrics(test)

        return metrics


def get_critical_samples(y_true_class_confidence: np.array,  # n_rows x 1
                         perturbed_df: pd.DataFrame,
                         top_k_samples: int = 5):
    if True: # Temporary
        return pd.DataFrame(), []
    # sparse format for the pair (orig_x, pert_x)
    stress_test_indicator = perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE]
    valid_stress_ids = set(stress_test_indicator) & set(DkuStressTestCenterConstants.FEATURE_PERTURBATIONS)
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
