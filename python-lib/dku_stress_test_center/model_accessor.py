# -*- coding: utf-8 -*-
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from dku_stress_test_center.utils import DkuStressTestCenterConstants

logger = logging.getLogger(__name__)

ALGORITHMS_WITH_VARIABLE_IMPORTANCE = [RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
                                       DecisionTreeClassifier]


class ModelAccessor(object):
    def __init__(self, model_handler=None):
        self.model_handler = model_handler

    def get_prediction_type(self):
        """
        Wrap the prediction type accessor of the model
        """
        if self.model_handler.get_prediction_type() in DkuStressTestCenterConstants.DKU_CLASSIFICATION_TYPE:
            return DkuStressTestCenterConstants.CLASSIFICATION_TYPE
        elif DkuStressTestCenterConstants.REGRESSION_TYPE in self.model_handler.get_prediction_type():
            return DkuStressTestCenterConstants.REGRESSION_TYPE
        else:
            return DkuStressTestCenterConstants.CLUSTERING_TYPE

    def get_target_variable(self):
        """
        Return the name of the target variable
        """
        return self.model_handler.get_target_variable()

    def get_original_test_df(self, limit=DkuStressTestCenterConstants.MAX_NUM_ROW):
        try:
            return self.model_handler.get_test_df()[0][:limit]
        except Exception as e:
            logger.warning(
                'Can not retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self.model_handler.get_full_df()[0][:limit]

    def get_per_feature(self):
        return self.model_handler.get_per_feature()

    def get_predictor(self):
        return self.model_handler.get_predictor()

    def predict(self, df):
        return self.get_predictor().predict(df)
