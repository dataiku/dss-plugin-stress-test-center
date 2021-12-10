# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_stress_test_center.metrics import Metric, worst_group_performance, performance_variation,\
    corruption_resilience_classification, corruption_resilience_regression
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants

class StressTest(object):
    def __init__(self, shift: Shift):
        self.shift = shift
        self.df_with_pred = None

    def perturb_df(self, df: pd.DataFrame):
        raise NotImplementedError()

    def compute_metrics(self, perf_metric: Metric, clean_y_true: np.array, perturbed_y_true: np.array,
                        clean_y_pred: np.array, perturbed_y_pred: np.array,
                        clean_probas: np.array, perturbed_probas: np.array):
        raise NotImplementedError()

    @property
    def name(self):
        return self.shift.__class__.__name__


class FeaturePerturbationTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.FEATURE_PERTURBATION

    def __init__(self, shift: Shift, features: list):
        super(FeaturePerturbationTest, self).__init__(shift)
        self.features = features

    def check(self, X):
        if self.shift.feature_type == PerturbationConstants.NUMERIC:
            if not pd.api.types.is_numeric_dtype(X):
                raise ValueError("Some selected features are not of a numeric type")
        if self.shift.feature_type in {PerturbationConstants.TEXT, PerturbationConstants.CATEGORICAL}:
            if not pd.api.types.is_string_dtype(X):
                raise ValueError("Some selected features are not of a string type")

    def perturb_df(self, df: pd.DataFrame):
        df = df.copy()
        X = df.loc[:, self.features]
        self.check(X)

        X, _ = self.shift.transform(X.values)
        df.loc[:, self.features] = X

        return df

    def compute_metrics(self, perf_metric: Metric, clean_y_true: np.array, perturbed_y_true: np.array,
                        clean_y_pred: np.array, perturbed_y_pred: np.array,
                        clean_probas: np.array, perturbed_probas: np.array):
        if perf_metric.pred_type == DkuStressTestCenterConstants.REGRESSION:
            corruption_resilience = corruption_resilience_regression(clean_y_pred, perturbed_y_pred, clean_y_true)
        else:
            corruption_resilience = corruption_resilience_classification(clean_y_pred, perturbed_y_pred)
        return {
            "corruption_resilience": corruption_resilience,
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, perturbed_y_true, clean_y_pred, perturbed_y_pred,
                clean_probas, perturbed_probas
            )
        }


class SubpopulationShiftTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.SUBPOPULATION_SHIFT

    def __init__(self, shift: Shift, population: str):
        super(SubpopulationShiftTest, self).__init__(shift)
        self.population = population

    def perturb_df(self, df: pd.DataFrame):
        df = df.copy()
        X = df.loc[:, df.columns != self.population].values
        y = df.loc[:, self.population].values

        X, y = self.shift.transform(X, y)
        df.loc[:, df.columns != self.population] = X
        df.loc[:, self.population] = y

        return df

    def compute_metrics(self, perf_metric: Metric, clean_y_true: np.array, perturbed_y_true: np.array,
                        clean_y_pred: np.array, perturbed_y_pred: np.array,
                        clean_probas: np.array, perturbed_probas: np.array):
        return {
            "worst_group_subpop": worst_group_performance(
                perf_metric, perturbed_y_true, perturbed_y_true, perturbed_y_pred, perturbed_probas
            ),
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, perturbed_y_true, clean_y_pred, perturbed_y_pred,
                clean_probas, perturbed_probas
            )
        }


class TargetShiftTest(SubpopulationShiftTest):
    TEST_TYPE = DkuStressTestCenterConstants.TARGET_SHIFT

    def compute_metrics(self, perf_metric: Metric, clean_y_true: np.array, perturbed_y_true: np.array,
                    clean_y_pred: np.array, perturbed_y_pred: np.array,
                    clean_probas: np.array, perturbed_probas: np.array):
        return {
            "performance_variation": performance_variation(
                perf_metric, clean_y_true, perturbed_y_true, clean_y_pred, perturbed_y_pred,
                clean_probas, perturbed_probas
            )
        }


class StressTestGenerator(object):
    def __init__(self, random_state=65537):
        self._random_state = random_state

        self.tests = None
        self.model_accessor = None
        self._sampling_proportion = None
        self._clean_df = None

    def generate_test(self, test_name, test_config):
        test_class, test_type = DkuStressTestCenterConstants.TESTS[test_name]
        params = test_config["params"]

        if test_type == DkuStressTestCenterConstants.FEATURE_PERTURBATION:
            features = test_config["selected_features"]
            return FeaturePerturbationTest(test_class(**params), features)

        if test_type == DkuStressTestCenterConstants.TARGET_SHIFT:
            target = self.model_accessor.get_target_variable()
            population = test_config.get("population", target)
            return TargetShiftTest(test_class(**params), population)

        raise ValueError("Unknown stress test %s" % test_name)

    def set_config(self, config: dict):
        self._sampling_proportion = config["samples"]
        self.random_state = config["randomSeed"]
        tests_config = config["perturbations"]
        self.tests = defaultdict(list)
        for test_name, test_config in tests_config.items():
            test = self.generate_test(test_name, test_config)
            self.tests[test.TEST_TYPE].append(test)

    def compute_test_metrics(self, test: StressTest):
        clean_df_with_pred = self._clean_df.loc[test.df_with_pred.index, :]

        target = self.model_accessor.get_target_variable()
        target_map = self.model_accessor.get_target_map()
        clean_y_true = clean_df_with_pred[target].replace(target_map)
        perturbed_y_true = test.df_with_pred[target].replace(target_map)
        clean_y_pred = clean_df_with_pred[DkuStressTestCenterConstants.PREDICTION].replace(target_map)
        perturbed_y_pred = test.df_with_pred[DkuStressTestCenterConstants.PREDICTION].replace(target_map)

        clean_probas = clean_df_with_pred.filter(regex=r'^proba_', axis=1).values
        perturbed_probas = test.df_with_pred.filter(regex=r'^proba_', axis=1).values


        return {
            test.name: test.compute_metrics(
                self.model_accessor.get_metric(), clean_y_true, perturbed_y_true,
                clean_y_pred, perturbed_y_pred, clean_probas, perturbed_probas
            )
        }

    def predict_clean_df(self, df: pd.DataFrame):
        self._clean_df = self.model_accessor.predict_and_concatenate(df.copy())

    def build_stress_metrics(self):
        metrics = defaultdict(lambda: {"metrics": {}})

        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
        self.predict_clean_df(df)

        for test_type, tests in self.tests.items():
            for test in tests:
                perturbed_df = test.perturb_df(df)
                test.df_with_pred = self.model_accessor.predict_and_concatenate(perturbed_df)
                metrics[test_type]["metrics"].update(self.compute_test_metrics(test))

        return metrics

    def _get_true_class_proba_columns(self, test_type: str):
        target = self.model_accessor.get_target_variable()
        true_probas_mask = pd.get_dummies(self._clean_df[target], prefix="proba", dtype=bool)

        uncorrupted_probas = self._clean_df[true_probas_mask.columns].values[true_probas_mask]
        true_class_probas = pd.DataFrame({
            DkuStressTestCenterConstants.UNCORRUPTED: uncorrupted_probas,
        }, index=true_probas_mask.index)

        for test in self.tests[test_type]:
            perturbed_probas = test.df_with_pred[true_probas_mask.columns]
            cropped_true_class_probas = true_probas_mask.loc[perturbed_probas.index, :]
            true_class_probas[test.name] = pd.Series(
                perturbed_probas.values[cropped_true_class_probas],
                index=perturbed_probas.index
            )

        return true_class_probas

    def _get_prediction_columns(self, test_type: str):
        uncorrupted_predictions = self._clean_df[DkuStressTestCenterConstants.PREDICTION]
        predictions = pd.DataFrame({
            DkuStressTestCenterConstants.UNCORRUPTED: uncorrupted_predictions
        })


        for test in self.tests[test_type]:
            predictions[test.name] = test.df_with_pred[DkuStressTestCenterConstants.PREDICTION]
        return predictions

    def get_critical_samples(self, test_type: str,
                             nr_samples: int = DkuStressTestCenterConstants.NR_CRITICAL_SAMPLES):
        if self.model_accessor.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION:
            columns = self._get_prediction_columns(test_type)
        else:
            columns = self._get_true_class_proba_columns(test_type)
        columns.dropna(inplace=True)
        if columns.empty:
            return {
                "uncertainties": [],
                "means": [],
                "samples": []
            }

        uncertainties = columns.std(axis=1)
        indexes_to_keep = uncertainties.nlargest(nr_samples).index

        critical_uncertainties = uncertainties.loc[indexes_to_keep]
        critical_means = columns.loc[indexes_to_keep, :].mean(axis=1)
        critical_samples = self.model_accessor.get_original_test_df(
            sample_fraction=self._sampling_proportion,
            random_state=self._random_state
        ).loc[indexes_to_keep, :]

        return {
            "uncertainties": critical_uncertainties.tolist(),
            "means": critical_means.tolist(),
            "samples": critical_samples.to_dict(orient='records')
        }
