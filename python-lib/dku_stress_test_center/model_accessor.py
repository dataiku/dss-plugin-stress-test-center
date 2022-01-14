# -*- coding: utf-8 -*-
import logging
import numpy as np
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dku_stress_test_center.metrics import Metric

logger = logging.getLogger(__name__)


class ModelAccessor(object):
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        return self.model_handler.get_prediction_type()

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_target_classes(self):
        if self.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION:
            return []
        return list(self.get_target_map().keys())

    def get_target_map(self):
        return self.model_handler.get_target_map()

    def get_original_test_df(self, sample_fraction, random_state):
        np.random.seed(random_state)
        try:
            test_df, _ = self.model_handler.get_test_df()
        except Exception as e:
            logger.warning(
                'Cannot retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            test_df, _ = self.model_handler.get_full_df()
        finally:
            return test_df.sample(frac=sample_fraction, random_state=random_state)

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_collector_data(self, feature):
        return self.model_handler.get_collector_data()["per_feature"][feature]

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def predict_and_concatenate(self, df):
        df_with_pred = self.model_handler.predict_and_concatenate(df)
        return df_with_pred.dropna(subset=[DkuStressTestCenterConstants.PREDICTION])

    @property
    def metrics(self):
        return self.model_handler.get_modeling_params()["metrics"]

    def get_evaluation_metric(self):
        initial_evaluation_metric = self.metrics["evaluationMetric"]
        if initial_evaluation_metric == Metric.CUSTOM:
            if self.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION:
                return Metric.R2
            return Metric.ROC_AUC
        return initial_evaluation_metric
