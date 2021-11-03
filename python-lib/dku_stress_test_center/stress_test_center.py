# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from drift_dac.perturbation_shared_utils import Shift
from sklearn.metrics import accuracy_score
from dku_webapp import DkuWebappConstants

class StressTest(object):
    def __init__(self, shift: Shift, list_of_features: list = None):
        self.shift = shift
        self.features = list_of_features
        self.perturbed_df = None
        self.sample_perturbation = self.__class__ in DkuStressTestCenterConstants.FEATURE_PERTURBATIONS
        # TODO: check valid configurations

    @staticmethod
    def create(shift_type, shift_params, shift_features):
        shift = DkuStressTestCenterConstants.TESTS[shift_type](**shift_params)
        return StressTest(shift, shift_features)

    def fit_transform(self, df: pd.DataFrame, target: str):
        self.perturbed_df = df

        if self.sample_perturbation:
            perturbed_columns = df.columns
        else:
            perturbed_columns = self.perturbed_df.columns != target

        xt = self.perturbed_df.loc[:, perturbed_columns].values
        yt = self.perturbed_df[target].values
        (xt, yt) = self.shift.transform(xt, yt)

        self.perturbed_df.loc[:, perturbed_columns] = xt
        self.perturbed_df.loc[:, target] = yt


class StressTestGenerator(object):
    def __init__(self, random_state=65537):
        self.tests = None
        self.model_accessor = None

        self._random_state = random_state
        self._sampling_proportion = None

    def performance_metric(self, y_true: np.array, y_pred: np.array):
        return accuracy_score(y_true, y_pred) # TODO

    def performance_variation(self, clean_y_true: np.array, clean_y_pred: np.array,
                              perturbed_y_true: np.array, perturbed_y_pred: np.array):
        clean_performance_metric = self.performance_metric(clean_y_true, clean_y_pred)
        stressed_performance_metric = self.performance_metric(perturbed_y_true, perturbed_y_pred)
        return clean_performance_metric - stressed_performance_metric # TODO

    def worst_group_performance(self, perturbed_y_true: np.array, perturbed_y_pred: np.array):
        target_classes = self.model_accessor.get_target_classes()
        performances = []
        for target_class in target_classes:
            class_mask = perturbed_y_true == target_class
            performances.append(self.performance_metric(perturbed_y_true[class_mask], perturbed_y_pred[class_mask]))
        return min(performances)

    def stress_resilience(self, clean_y_pred: np.array, perturbed_y_pred: np.array):
        # TODO: make it work for regression as well
        return (clean_y_pred == perturbed_y_pred).sum() / len(clean_y_pred)

    def compute_metrics(self, test: StressTest, clean_y_true: np.array, clean_y_pred: np.array,
                        perturbed_y_true: np.array, perturbed_y_pred: np.array):
        if test.sample_perturbation:
            robustness = self.stress_resilience(clean_y_pred, perturbed_y_pred)
        else:
            robustness = self.worst_group_performance(perturbed_y_true, perturbed_y_pred)
        return {
            "performance_variation": self.performance_variation(clean_y_true, clean_y_pred,
                                                                perturbed_y_true,
                                                                perturbed_y_pred),
            "robustness": robustness
        }

    def set_config(self, config: dict):
        self._sampling_proportion = config["samples"]
        tests_config = config["perturbations"]
        self.tests = []
        for shift_type, shift_config in tests_config.items():
            params, features = shift_config["params"], shift_config.get("selected_features")
            self.tests.append(StressTest.create(shift_type, params, features))

    def fit_transform(self):
        target = self.model_accessor.get_target_variable()

        for test in self.tests:
            df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                            random_state=self._random_state)
            test.fit_transform(df, target)

    def build_stress_metric(self):
        metrics = {
            DkuWebappConstants.SAMPLE_PERTURBATION: {"metrics": {}},
            DkuWebappConstants.SUBPOPULATION_PERTURBATION: {"metrics": {}}
        }
        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                        random_state=self._random_state)
        target = self.model_accessor.get_target_variable()
        clean_y_true = df[target]
        clean_y_pred = self.model_accessor.predict(df)[DkuWebappConstants.PREDICTION]
        for test in self.tests:
            perturbed_y_true = test.perturbed_df[target]
            perturbed_y_pred = self.model_accessor.predict(test.perturbed_df)[DkuWebappConstants.PREDICTION]
            test_type = DkuWebappConstants.SAMPLE_PERTURBATION if test.sample_perturbation\
                    else DkuWebappConstants.SUBPOPULATION_PERTURBATION
            metrics[test_type]["metrics"].update({
                test.shift.__class__.__name__:
                    self.compute_metrics(test, clean_y_true, clean_y_pred, perturbed_y_true,
                                         perturbed_y_pred)
            })
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
