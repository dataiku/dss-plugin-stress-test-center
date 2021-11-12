# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_stress_test_center.metrics import worst_group_performance, performance_variation, stress_resilience
from drift_dac.perturbation_shared_utils import Shift

class StressTest(object):
    def __init__(self, shift: Shift):
        self.shift = shift
        self.df_with_pred = None
        # TODO: check valid configurations

    @staticmethod
    def create(test_name, test_config):
        test_class, test_type = DkuStressTestCenterConstants.TESTS[test_name]
        params = test_config["params"]
        if test_type == DkuStressTestCenterConstants.FEATURE_PERTURBATION:
            feature_perturbation = test_class(**params)
            features = test_config["selected_features"]
            return FeaturePerturbationTest(feature_perturbation, features)
        if test_type == DkuStressTestCenterConstants.SUBPOPULATION_SHIFT:
            subpopulation_shift = test_class(**params)
            return SubpopulationShiftTest(subpopulation_shift)
        raise ValueError("Unknown stress test %s" % test_name)

    def perturb_df(self, df: pd.DataFrame, target: str):
        raise NotImplementedError()

    def compute_metrics(self, perf_metric: str, clean_y_true: np.array, clean_y_pred: np.array,
                        perturbed_y_true: np.array, perturbed_y_pred: np.array):
        raise NotImplementedError()


class FeaturePerturbationTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.FEATURE_PERTURBATION

    def __init__(self, shift: Shift, features: list):
        super(FeaturePerturbationTest, self).__init__(shift)
        self.features = features

    def perturb_df(self, df: pd.DataFrame, target: str):
        X = df.loc[:, self.features].values
        y = df.loc[:, target].values

        X, y = self.shift.transform(X, y)
        df.loc[:, self.features] = X
        df.loc[:, target] = y

        return df

    def compute_metrics(self, perf_metric: str, clean_y_true: np.array, clean_y_pred: np.array,
                        perturbed_y_true: np.array, perturbed_y_pred: np.array):
        return {
            "robustness": stress_resilience(clean_y_pred, perturbed_y_pred),
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, clean_y_pred, perturbed_y_true, perturbed_y_pred
            )
        }


class SubpopulationShiftTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.SUBPOPULATION_SHIFT

    def perturb_df(self, df: pd.DataFrame, target: str):
        X = df.loc[:, df.columns != target].values
        y = df.loc[:, target].values

        X, y = self.shift.transform(X, y)
        df.loc[:, df.columns != target] = X
        df.loc[:, target] = y

        return df

    def compute_metrics(self, perf_metric: str, clean_y_true: np.array, clean_y_pred: np.array,
                        perturbed_y_true: np.array, perturbed_y_pred: np.array):
        return {
            "robustness": worst_group_performance(perf_metric, perturbed_y_true, perturbed_y_pred),
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, clean_y_pred, perturbed_y_true, perturbed_y_pred
            )
        }


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
        self.tests = defaultdict(list)
        for test_name, test_config in tests_config.items():
            test = StressTest.create(test_name, test_config)
            self.tests[test.TEST_TYPE].append(test)

    def compute_test_metrics(self, test: StressTest):
        target = self.model_accessor.get_target_variable()
        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
        perturbed_df = test.perturb_df(df, target)
        test.df_with_pred = self.model_accessor.predict_and_concatenate(perturbed_df)
        clean_df_with_pred = self._clean_df.loc[test.df_with_pred.index, :]

        clean_y_true = clean_df_with_pred[target]
        clean_y_pred = clean_df_with_pred[DkuStressTestCenterConstants.PREDICTION]
        perturbed_y_true = test.df_with_pred[target]
        perturbed_y_pred = test.df_with_pred[DkuStressTestCenterConstants.PREDICTION]

        return {
            test.shift.__class__.__name__: test.compute_metrics(
                "auc", clean_y_true, clean_y_pred, perturbed_y_true, perturbed_y_pred
            ) # TODO: use get_evaluation_metric instead of hardcoded auc
        }

    def predict_clean_df(self):
        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
        self._clean_df = self.model_accessor.predict_and_concatenate(df)

    def build_stress_metrics(self):
        metrics = defaultdict(lambda: {"metrics": {}})

        self.predict_clean_df()

        for test_type, tests in self.tests.items():
            for test in tests:
                metrics[test_type]["metrics"].update(self.compute_test_metrics(test))

        return metrics


    def get_critical_samples(self, test_type: str,
                             nr_samples: int = DkuStressTestCenterConstants.NR_CRITICAL_SAMPLES):

        target = self.model_accessor.get_target_variable()
        true_probas_mask = pd.get_dummies(self._clean_df[target], prefix="proba", dtype=bool)
        true_class_probas = pd.DataFrame({
            0: self._clean_df[true_probas_mask.columns].values[true_probas_mask]
        })

        for idx, test in enumerate(self.tests[test_type]):
            perturbed_probas = test.df_with_pred[true_probas_mask.columns]
            cropped_true_class_probas = true_probas_mask.loc[perturbed_probas.index, :]
            true_class_probas[idx+1] = pd.Series(
                perturbed_probas.values[cropped_true_class_probas],
                index=perturbed_probas.index
            )
        true_class_probas.dropna(inplace=True, how='all')
        uncertainties = true_class_probas.std(axis=1)
        indexes_to_keep = uncertainties.nlargest(nr_samples).index

        if indexes_to_keep.empty:
            return {
                "uncertainties": [],
                "mean_true_proba": [],
                "features": []
            }

        critical_uncertainties = uncertainties.loc[indexes_to_keep]
        critical_true_proba_means = true_class_probas.loc[indexes_to_keep, :].mean(axis=1)
        critical_samples = self.model_accessor.get_original_test_df(
            sample_fraction=self._sampling_proportion,
            random_state=self._random_state
        ).loc[indexes_to_keep, :]

        return {
            "uncertainties": critical_uncertainties.tolist(),
            "mean_true_proba": critical_true_proba_means.tolist(),
            "features": critical_samples.to_dict(orient='records')
        }
