# -*- coding: utf-8 -*-
import dataiku

from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import build_stress_metric, get_critical_samples
from dku_stress_test_center.stress_test_center import DkuStressTestCenterConstants
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial
from model_metadata import get_model_handler
from drift_dac.prior_shift import KnockOut

import numpy as np
import logging
import simplejson
import traceback
from dku_model_accessor import get_model_handler, ModelAccessor, DkuModelAccessorConstants
from dku_webapp import remove_nan_from_list, convert_numpy_int64_to_int, get_metrics, get_histograms,DkuWebappConstants

logger = logging.getLogger(__name__)


@app.route('/compte/<model_id>/<version_id>')
def compute(model_id, version_id):
    try:
        print('Compute starts ...')

        return simplejson.dumps({})

        model = dataiku.Model(model_id)
        model_handler = get_model_handler(model=model, version_id=version_id)
        model_accessor = ModelAccessor(model_handler)

        # Get test data
        logger.info('Retrieving model data...')
        test_df = model_accessor.get_original_test_df()
        target = model_accessor.get_target_variable()

        def undo_preproc_name(f):
            if ':' in f:
                return f.split(':')[1]
            else:
                return f

        selected_features = set()
        feature_importance = model_accessor.get_feature_importance().index.tolist()

        for feature in feature_importance:
            selected_features.add(undo_preproc_name(feature))
            if len(selected_features) > 10:
                break

        selected_features = list(selected_features)
        logger.info('List of selected features for the stress test: ', selected_features)
        is_categorical = [False] * len(selected_features)
        is_text = [False] * len(selected_features)

        # Run the stress tests
        config_list = [StressTestConfiguration(KnockOut()),
                       StressTestConfiguration(MissingValues()),
                       StressTestConfiguration(Scaling()),
                       StressTestConfiguration(Adversarial())]

        stressor = StressTestGenerator(config_list, selected_features, is_categorical, is_text)
        perturbed_df = stressor.fit_transform(test_df)  # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_

        perturbed_df_with_prediction = model_accessor.predict(perturbed_df)

        # Compute the performance drop metrics
        metrics_df = build_stress_metric(y_true=perturbed_df[target],
                                         y_pred=perturbed_df_with_prediction['prediction'],
                                         stress_test_indicator=perturbed_df[
                                             DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                         pos_label='>50K')

        y_true = perturbed_df[target]
        y_true_class_confidence = perturbed_df_with_prediction[['proba_<=50K', 'proba_>50K']].values
        y_true_idx = np.array([[True, False] if y == '>=50K' else [False, True] for y in y_true])
        y_true_class_confidence = y_true_class_confidence[y_true_idx]

        critical_samples_df = get_critical_samples(y_true_class_confidence=y_true_class_confidence,
                                                   stress_test_indicator=perturbed_df[
                                                       DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                                   row_indicator=perturbed_df[DkuStressTestCenterConstants.DKU_ROW_ID])

        data = {
            'metrics': metrics_list,
            'critical_samples': critical_samples_list
        }
        return simplejson.dumps(data, ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500

@app.route('/get-outcome-list/<model_id>/<version_id>')
def get_outcome_list(model_id, version_id):
    try:
        model = dataiku.Model(model_id)
        model_handler = get_model_handler(model, version_id=version_id)
        model_accessor = ModelAccessor(model_handler)
        # note: sometimes when the dataset is very unbalanced, the original_test_df does not have all the target values
        test_df = model_accessor.get_original_test_df()
        target = model_accessor.get_target_variable()
        outcome_list = test_df[target].unique().tolist()
        filtered_outcome_list = remove_nan_from_list(outcome_list)
        return simplejson.dumps(filtered_outcome_list, ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call get-outcome-list endpoint: {}.".format(traceback.format_exc()))
        return "{}Check backend log for more details.".format(traceback.format_exc()), 500

@app.route('/get-data/<model_id>/<version_id>/<advantageous_outcome>/<sensitive_column>/<reference_group>')
def get_data(model_id, version_id, advantageous_outcome, sensitive_column, reference_group):
    try:
        if sensitive_column == 'undefined' or sensitive_column == 'null':
            raise ValueError('Please choose a column.')
        if reference_group == 'undefined' or reference_group == 'null':
            raise ValueError('Please choose a sensitive group.')
        if  advantageous_outcome == 'undefined' or advantageous_outcome == 'null':
            raise ValueError('Please choose an outcome.')

        populations, disparity_dct, label_list = get_metrics(model_id, version_id, advantageous_outcome, sensitive_column, reference_group)
        histograms = get_histograms(model_id, version_id, advantageous_outcome, sensitive_column)
        # the following strings are used only here, too lazy to turn them into constant variables
        data = {'populations': populations,
                'histograms': histograms,
                'disparity': disparity_dct,
                'labels': label_list
                }
        return simplejson.dumps(data, ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call get-data endpoint: {}.".format(traceback.format_exc()))
        return "{}Check backend log for more details.".format(traceback.format_exc()), 500