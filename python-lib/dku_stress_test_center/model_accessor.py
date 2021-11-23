# -*- coding: utf-8 -*-
import logging
import numpy as np
from dku_stress_test_center.utils import DkuStressTestCenterConstants

logger = logging.getLogger(__name__)


class ModelAccessor(object):
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        if self.model_handler.get_prediction_type() in DkuStressTestCenterConstants.DKU_CLASSIFICATION_TYPE:
            return DkuStressTestCenterConstants.CLASSIFICATION_TYPE
        if DkuStressTestCenterConstants.REGRESSION_TYPE in self.model_handler.get_prediction_type():
            return DkuStressTestCenterConstants.REGRESSION_TYPE
        return DkuStressTestCenterConstants.CLUSTERING_TYPE

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_target_classes(self):
        if self.get_prediction_type() == DkuStressTestCenterConstants.REGRESSION_TYPE:
            return []
        return list(self.model_handler.get_target_map().keys())

    def get_evaluation_metric(self):
        return self.model_handler.get_modeling_params["metrics"]["evaluationMetric"]

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

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def predict_and_concatenate(self, df):
        df_with_pred = self.model_handler.predict_and_concatenate(df)
        return df_with_pred.dropna(subset=[DkuStressTestCenterConstants.PREDICTION])
