# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
from sklearn.exceptions import UndefinedMetricWarning

from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_webapp import MISSING_VALUE, safe_str
from dku_stress_test_center.metrics import Metric, worst_group_performance,\
    corruption_resilience_classification, corruption_resilience_regression

from drift_dac.perturbation_shared_utils import PerturbationConstants
from drift_dac.covariate_shift import MissingValues, Scaling

warnings.filterwarnings("error", category=UndefinedMetricWarning)

class StressTest(object):
    def __init__(self, test_name, params):
        self.shift = DkuStressTestCenterConstants.TESTS[test_name](params)
        self.name = test_name

        self.df_with_pred = None
        self.not_relevant_explanation = None
        self.y_true = None
        self.y_pred = None
        self.probas = None
        self.sample_weights = None

    def perturb_df(self, df):
        raise NotImplementedError()

    @property
    def relevant(self):
        return not bool(self.not_relevant_explanation)

    def check_relevance(self, preprocessing):
        pass

    def compute_specific_metrics(self, metric, clean_y_true, clean_y_pred):
        raise NotImplementedError()


class FeaturePerturbationTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.FEATURE_PERTURBATION
    TESTS = {"MissingValues", "Scaling"}

    def __init__(self, test_name, params, selected_features):
        super(FeaturePerturbationTest, self).__init__(test_name, params)
        self.features = selected_features

    def check_relevance(self, preprocessing):
        if type(self.shift) is Scaling:
            if self.shift.scaling_factor == 1:
                self.not_relevant_explanation = "The scaling factor is set to 1."
        elif type(self.shift) is MissingValues:
            for feature in self.features:
                if preprocessing[feature].get("missing_handling", "") == "DROP_ROW":
                    self.not_relevant_explanation = ("The feature '{}' drops ".format(feature) +\
                        "the rows with missing values and does not predict them.")
        else:
            raise ValueError("Wrong feature corruption test class: {}".format(type(self.shift)))

    def _check_proper_column_types(self, df):
        for feature in df:
            if self.shift.feature_type == PerturbationConstants.NUMERIC:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    raise ValueError("{} is not of a numeric type".format(feature))
            if self.shift.feature_type in {PerturbationConstants.TEXT, PerturbationConstants.CATEGORICAL}:
                if not pd.api.types.is_string_dtype(df[feature]):
                    raise ValueError("{} is not of a string type".format(feature))

    def perturb_df(self, df):
        df = df.copy()
        X = df.loc[:, self.features]
        self._check_proper_column_types(X)

        X, _ = self.shift.transform(X.values)
        df.loc[:, self.features] = X

        return df

    def compute_specific_metrics(self, metric, clean_y_true, clean_y_pred):
        if not self.relevant:
            # Altered and unaltered datasets are the same, including the prediction columns.
            # By definition, corruption resilience will hence always be 1.
            corruption_resilience = 1
        elif metric.pred_type == DkuStressTestCenterConstants.REGRESSION:
            corruption_resilience = corruption_resilience_regression(
                clean_y_pred, self.y_pred, clean_y_true
            )
        else:
            corruption_resilience = corruption_resilience_classification(
                clean_y_pred, self.y_pred
            )
        return [{
            "value": corruption_resilience,
            "name": "corruption_resilience"
        }]


class SubpopulationShiftTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.SUBPOPULATION_SHIFT
    TESTS = {"RebalanceFeature"}

    def __init__(self, test_name, params, population):
        super(SubpopulationShiftTest, self).__init__(test_name, params)
        self.population = population

    def perturb_df(self, df):
        df = df.copy()
        X = df.loc[:, df.columns != self.population].values
        y = df.loc[:, self.population].replace({np.nan: MISSING_VALUE}).values

        X, y = self.shift.transform(X, y)
        df.loc[:, df.columns != self.population] = X
        df.loc[:, self.population] = y

        return df

    def compute_specific_metrics(self, metric, clean_y_true, clean_y_pred):
        worst_subpop_perf_dict = {"name": "worst_subpop_perf"}
        subpopulation = self.df_with_pred.loc[self.y_true.index, self.population]
        try:
            worst_group_perf = worst_group_performance(
                metric, subpopulation, self.y_true,
                self.y_pred, self.probas, self.sample_weights
            )
        except:
            worst_subpop_perf_dict["warning"] = metric.name + " is ill-defined for " +\
                "some modalities. Fell back to using accuracy."
            metric = Metric(Metric.ACCURACY)
            worst_group_perf = worst_group_performance(
                metric, subpopulation, self.y_true,
                self.y_pred, self.probas, self.sample_weights
            )
        ret = {
            "base_metric": metric.name,
            "value": worst_group_perf,
        }
        ret.update(worst_subpop_perf_dict)
        return [ret]


class TargetShiftTest(SubpopulationShiftTest):
    TEST_TYPE = DkuStressTestCenterConstants.TARGET_SHIFT
    TESTS = {"RebalanceTarget"}

    def compute_specific_metrics(self, metric, clean_y_true, clean_y_pred):
        return [] 


class StressTestGenerator(object):
    def __init__(self):
        self.model_accessor = None

        self._metric = None
        self._random_state = None
        self._tests = None
        self._sampling_proportion = None
        self._clean_df = None

    def generate_test(self, test_name, test_config):
        if test_name in FeaturePerturbationTest.TESTS:
            return FeaturePerturbationTest(test_name, **test_config)
        if test_name in TargetShiftTest.TESTS:
            test_config["population"] = self.model_accessor.get_target_variable()
            return TargetShiftTest(test_name, **test_config)
        if test_name in SubpopulationShiftTest.TESTS:
            return SubpopulationShiftTest(test_name, **test_config)
        raise ValueError("Unknown stress test %s" % test_name)

    def set_config(self, config):
        self._sampling_proportion = config["samples"]
        self._random_state = config["randomSeed"]
        self._metric = Metric(config["perfMetric"], self.model_accessor.metrics,
                              self.model_accessor.get_prediction_type())

        self._tests = defaultdict(list)
        tests = config["tests"]
        feature_preprocessing = self.model_accessor.get_per_feature()
        for test_name, test_config in tests.items():
            test = self.generate_test(test_name, test_config)
            test.check_relevance(feature_preprocessing)
            self._tests[test.TEST_TYPE].append(test)

    def _get_col_for_metrics(self, df):
        target = self.model_accessor.get_target_variable()
        target_map = self.model_accessor.get_target_map()
        weight_var = self.model_accessor.get_weight_variable()

        y_true = df[target].replace(target_map)
        y_pred = df[DkuStressTestCenterConstants.PREDICTION].replace(target_map)
        probas = df.filter(regex=r'^proba_', axis=1).values
        sample_weights = df[weight_var] if weight_var else None
        return y_true, y_pred, probas, sample_weights

    def compute_test_metrics(self, test):
        per_test_metrics = {"name": test.name}

        perf_after_dict = {"name": "perf_after"}
        test.y_true, test.y_pred, test.probas, test.sample_weights = self._get_col_for_metrics(test.df_with_pred)
        metric = self._metric
        try:
            perf_after = self._metric.compute(test.y_true, test.y_pred, test.probas, test.sample_weights)
        except:
            metric = Metric(Metric.ACCURACY)
            per_test_metrics["warning"] = self._metric.name + " was ill-defined for the altered dataset. "+\
                "Fell back to using accuracy."
            perf_after = metric.compute(test.y_true, test.y_pred, test.probas, test.sample_weights)
        perf_after_dict["base_metric"] = metric.name
        perf_after_dict["value"] = perf_after

        clean_y_true, clean_y_pred, clean_probas, clean_sample_weights = self._get_col_for_metrics(self._clean_df)
        if test.relevant:
            try:
                perf_before = metric.compute(clean_y_true, clean_y_pred, clean_probas, clean_sample_weights)
            except Exception as e:
                raise Exception("Failed to compute the performance (%s) on the unaltered test set: %s"
                    % (self._metric.name, safe_str(e)))
        else:
            # Altered and unaltered datasets are the same, including the prediction columns.
            # By definition, the performance is the same before and after the stress test.
            perf_before = perf_after
            per_test_metrics["warning"] = "Not relevant test: " + test.not_relevant_explanation

        common_metrics = [
            {
                "name": "perf_before",
                "value": perf_before,
                "base_metric": metric.name
            },
            perf_after_dict,
            {
                "name": "perf_var",
                "value": (perf_before - perf_after) * (-1 if metric.is_greater_better() else 1),
                "base_metric": metric.name
            }
        ]

        extra_metrics = test.compute_specific_metrics(metric, clean_y_true, clean_y_pred)
        per_test_metrics["metrics"] = common_metrics + extra_metrics
        return per_test_metrics

    def build_results(self):
        results = {}

        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)

        if self.model_accessor.get_prediction_type() != DkuStressTestCenterConstants.REGRESSION:
            target_map = self.model_accessor.get_target_map()
            target = self.model_accessor.get_target_variable()
            df = df[df[target].isin(target_map)]
        
        self._clean_df = self.model_accessor.predict_and_concatenate(df)
        if self._clean_df.shape[0] == 0:
            raise ValueError(
                "The test dataset is empty"
            )

        for test_type, tests in self._tests.items():
            results[test_type] = {"per_test": []}
            for test in tests:
                perturbed_df = test.perturb_df(df)
                test.df_with_pred = self.model_accessor.predict_and_concatenate(perturbed_df)
                if test.df_with_pred.shape[0] == 0:
                    raise ValueError(
                        "The test dataset is empty after applying the stress test '" +\
                        DkuStressTestCenterConstants.TEST_NAMES[test.name] + "'"
                    )

                results[test_type]["per_test"].append(self.compute_test_metrics(test))

        return results

    def _get_true_class_proba_columns(self, test_type):
        target = self.model_accessor.get_target_variable()
        true_probas_mask = pd.get_dummies(self._clean_df[target], prefix="proba", dtype=bool)

        uncorrupted_probas = self._clean_df[true_probas_mask.columns].values[true_probas_mask]
        true_class_probas = pd.DataFrame({
            DkuStressTestCenterConstants.UNCORRUPTED: uncorrupted_probas,
        }, index=true_probas_mask.index)

        for test in self._tests[test_type]:
            perturbed_probas = test.df_with_pred[true_probas_mask.columns]
            cropped_true_class_probas = true_probas_mask.loc[perturbed_probas.index, :]
            true_class_probas[test.name] = pd.Series(
                perturbed_probas.values[cropped_true_class_probas],
                index=perturbed_probas.index
            )

        return true_class_probas

    def _get_prediction_columns(self, test_type):
        uncorrupted_predictions = self._clean_df[DkuStressTestCenterConstants.PREDICTION]
        predictions = pd.DataFrame({
            DkuStressTestCenterConstants.UNCORRUPTED: uncorrupted_predictions
        })

        for test in self._tests[test_type]:
            predictions[test.name] = test.df_with_pred[DkuStressTestCenterConstants.PREDICTION]
        return predictions

    def get_critical_samples(self, test_type,
                             nr_samples=DkuStressTestCenterConstants.NR_CRITICAL_SAMPLES):
        if self.model_accessor.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION:
            columns = self._get_prediction_columns(test_type)
        else:
            columns = self._get_true_class_proba_columns(test_type)
        columns.dropna(inplace=True)
        if columns.empty:
            return {
                "uncertainties": [],
                "means": [],
                "samples": [],
                "predList": []
            }

        uncertainties = columns.std(axis=1)
        indexes_to_keep = uncertainties.nlargest(nr_samples).index

        critical_uncertainties = uncertainties.loc[indexes_to_keep]
        critical_means = columns.loc[indexes_to_keep, :].mean(axis=1)
        critical_samples = self.model_accessor.get_original_test_df(
            sample_fraction=self._sampling_proportion,
            random_state=self._random_state
        ).loc[indexes_to_keep, :].replace({np.nan: ""})

        return {
            "uncertainties": critical_uncertainties,
            "means": critical_means,
            "samples": critical_samples,
            "predList": columns.loc[indexes_to_keep, :]
        }
