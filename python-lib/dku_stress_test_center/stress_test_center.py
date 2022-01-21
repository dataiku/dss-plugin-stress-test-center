# -*- coding: utf-8 -*-
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.exceptions import UndefinedMetricWarning

from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_stress_test_center.metrics import Metric, worst_group_performance,\
    corruption_resilience_classification, corruption_resilience_regression

from drift_dac.perturbation_shared_utils import PerturbationConstants
from drift_dac.covariate_shift import MissingValues, Scaling

warnings.filterwarnings("error", category=UndefinedMetricWarning)

class StressTest(object):
    def __init__(self, test_name: str, params: dict):
        self.shift = DkuStressTestCenterConstants.TESTS[test_name](params)
        self.name = test_name
        self.df_with_pred = None
        self.not_relevant_explanation = None

    def perturb_df(self, df: pd.DataFrame):
        raise NotImplementedError()

    @property
    def relevant(self):
        return not bool(self.not_relevant_explanation)

    def check_relevance(self, preprocessing: dict):
        pass

class FeaturePerturbationTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.FEATURE_PERTURBATION
    TESTS = {"MissingValues", "Scaling"}

    def __init__(self, test_name: str, params: dict, selected_features: list):
        super(FeaturePerturbationTest, self).__init__(test_name, params)
        self.features = selected_features

    def check_relevance(self, preprocessing: dict):
        if type(self.shift) is Scaling:
            if self.shift.scaling_factor == 1:
                self.not_relevant_explanation = "The scaling factor is set to 1."
        elif type(self.shift) is MissingValues:
            for feature in self.features:
                if preprocessing[feature]["missing_handling"] == "DROP_ROW":
                    self.not_relevant_explanation = ("The feature '{}' drops ".format(feature) +\
                        "the rows with missing values and does not predict them.")
        else:
            raise ValueError("Wrong feature corruption test class: {}".format(type(self.shift)))

    def _check_proper_column_types(self, df: pd.DataFrame):
        for feature in df:
            if self.shift.feature_type == PerturbationConstants.NUMERIC:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    raise ValueError("{} is not of a numeric type".format(feature))
            if self.shift.feature_type in {PerturbationConstants.TEXT, PerturbationConstants.CATEGORICAL}:
                if not pd.api.types.is_string_dtype(df[feature]):
                    raise ValueError("{} is not of a string type".format(feature))

    def perturb_df(self, df: pd.DataFrame):
        df = df.copy()
        X = df.loc[:, self.features]
        self._check_proper_column_types(X)

        X, _ = self.shift.transform(X.values)
        df.loc[:, self.features] = X

        return df


class SubpopulationShiftTest(StressTest):
    TEST_TYPE = DkuStressTestCenterConstants.SUBPOPULATION_SHIFT
    TESTS = {"RebalanceFeature"}

    def __init__(self, test_name: str, params: dict, population: str):
        super(SubpopulationShiftTest, self).__init__(test_name, params)
        self.population = population

    def perturb_df(self, df: pd.DataFrame):
        df = df.copy()
        X = df.loc[:, df.columns != self.population].values
        y = df.loc[:, self.population].values

        X, y = self.shift.transform(X, y)
        df.loc[:, df.columns != self.population] = X
        df.loc[:, self.population] = y

        return df


class TargetShiftTest(SubpopulationShiftTest):
    TEST_TYPE = DkuStressTestCenterConstants.TARGET_SHIFT
    TESTS = {"RebalanceTarget"}


class StressTestGenerator(object):
    def __init__(self):
        self.model_accessor = None

        self._metric = None
        self._random_state = None
        self._tests = None
        self._sampling_proportion = None
        self._clean_df = None

    def generate_test(self, test_name: str, test_config: dict):
        if test_name in FeaturePerturbationTest.TESTS:
            return FeaturePerturbationTest(test_name, **test_config)
        if test_name in TargetShiftTest.TESTS:
            test_config["population"] = self.model_accessor.get_target_variable()
            return TargetShiftTest(test_name, **test_config)
        if test_name in SubpopulationShiftTest.TESTS:
            return SubpopulationShiftTest(test_name, **test_config)
        raise ValueError("Unknown stress test %s" % test_name)

    def set_config(self, config: dict):
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

    def _get_col_for_metrics(self, df: pd.DataFrame):
        target = self.model_accessor.get_target_variable()
        target_map = self.model_accessor.get_target_map()

        y_true = df[target].replace(target_map)
        y_pred = df[DkuStressTestCenterConstants.PREDICTION].replace(target_map)
        probas = df.filter(regex=r'^proba_', axis=1).values
        return y_true, y_pred, probas

    def compute_test_metrics(self, test: StressTest, clean_df_with_pred: pd.DataFrame):
        per_test_metrics = {
            "name": test.name
        }

        clean_y_true, clean_y_pred, clean_probas = self._get_col_for_metrics(clean_df_with_pred)
        perf_before = self._metric.compute(clean_y_true, clean_y_pred, clean_probas)
        per_test_metrics["metrics"] = [{
            "name": "perf_before",
            "value": perf_before,
            "base_metric": self._metric.name
        }]

        if test.relevant:
            perturbed_y_true, perturbed_y_pred, perturbed_probas = self._get_col_for_metrics(test.df_with_pred)
            perf_after = self._metric.compute(perturbed_y_true, perturbed_y_pred, perturbed_probas)
        else:
            # Altered and unaltered datasets are the same, including the prediction columns.
            # By definition, the performance is the same before and after the stress test.
            perf_after = perf_before
            per_test_metrics["not_relevant_explanation"] = test.not_relevant_explanation

        per_test_metrics["metrics"] += [
            {
                "name": "perf_after",
                "value": perf_after,
                "base_metric": self._metric.name
            },
            {
                "name": "perf_var",
                "value": (perf_before - perf_after) * (-1 if self._metric.is_greater_better() else 1),
                "base_metric": self._metric.name
            }
        ]

        if test.TEST_TYPE == DkuStressTestCenterConstants.FEATURE_PERTURBATION:
            if not test.relevant:
                # Altered and unaltered datasets are the same, including the prediction columns.
                # By definition, corruption resilience will hence always be 1.
                corruption_resilience = 1
            elif self.model_accessor.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION:
                corruption_resilience = corruption_resilience_regression(
                    clean_y_pred, perturbed_y_pred, clean_y_true
                )
            else:
                corruption_resilience = corruption_resilience_classification(
                    clean_y_pred, perturbed_y_pred
                )
            per_test_metrics["metrics"].append({"name": "corruption_resilience", "value": corruption_resilience})

        elif test.TEST_TYPE == DkuStressTestCenterConstants.SUBPOPULATION_SHIFT:
            worst_subpop_perf_dict = {"name": "worst_subpop_perf"}
            try:
                metric = self._metric
                worst_group_perf = worst_group_performance(
                    metric, test.df_with_pred[test.population], perturbed_y_true,
                    perturbed_y_pred, perturbed_probas
                )
            except:
                worst_subpop_perf_dict["warning"] = self._metric.name + " is ill-defined for " +\
                    "some modalities. Fell back to using accuracy."
                metric = Metric(Metric.ACCURACY)
                worst_group_perf = worst_group_performance(
                    metric, test.df_with_pred[test.population], perturbed_y_true,
                    perturbed_y_pred, perturbed_probas
                )
            worst_subpop_perf_dict["base_metric"] = metric.name
            worst_subpop_perf_dict["value"] = worst_group_perf
            per_test_metrics["metrics"].append(worst_subpop_perf_dict)

        return per_test_metrics

    def build_results(self):
        results = {}

        df = self.model_accessor.get_original_test_df(sample_fraction=self._sampling_proportion,
                                                      random_state=self._random_state)
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
                        "The test dataset is empty after applying the stress test " +\
                        DkuStressTestCenterConstants.FRIENDLY_NAMES[test.name]
                    )

                clean_df_with_pred = self._clean_df
                if test.df_with_pred.shape[0] < clean_df_with_pred.shape[0]:
                    clean_df_with_pred = clean_df_with_pred.loc[test.df_with_pred.index, :]
                results[test_type]["per_test"].append(self.compute_test_metrics(test, clean_df_with_pred))

        return results

    def _get_true_class_proba_columns(self, test_type: str):
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

    def _get_prediction_columns(self, test_type: str):
        uncorrupted_predictions = self._clean_df[DkuStressTestCenterConstants.PREDICTION]
        predictions = pd.DataFrame({
            DkuStressTestCenterConstants.UNCORRUPTED: uncorrupted_predictions
        })

        for test in self._tests[test_type]:
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
        ).loc[indexes_to_keep, :]

        return {
            "uncertainties": critical_uncertainties.tolist(),
            "means": critical_means.tolist(),
            "samples": critical_samples.to_dict(orient='records'),
            "predList": columns.loc[indexes_to_keep, :].to_dict(orient='records')
        }
