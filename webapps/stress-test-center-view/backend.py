# -*- coding: utf-8 -*-
import numpy as np
import logging
import simplejson
import traceback
from flask import request

from dataiku import api_client, Model
from dataiku.customwebapp import get_webapp_config
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import build_stress_metric, get_critical_samples
from dku_stress_test_center.stress_test_center import DkuStressTestCenterConstants
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos
from model_metadata import get_model_handler
from drift_dac.prior_shift import KnockOut
from dku_webapp import convert_numpy_int64_to_int, pretty_floats

logger = logging.getLogger(__name__)


@app.route('/compute/<model_id>/<version_id>')
def compute(model_id, version_id):
    try:

        fmi = get_webapp_config().get("trainedModelFullModelId")
        if fmi is None:
            model = Model(get_webapp_config()["modelId"])
            version_id = get_webapp_config().get("versionId")
            original_model_handler = get_model_handler(model, version_id)
            model_accessor = ModelAccessor(original_model_handler)
        else:
            original_model_handler = PredictionModelInformationHandler.from_full_model_id(fmi)
            model_accessor = ModelAccessor(original_model_handler)

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
        logger.info('List of selected features for the stress test: {}'.format(selected_features))

        feature_handling_dict = model_accessor.get_per_feature()
        is_text = []
        is_categorical = []
        for feature in selected_features:
            feature_params = feature_handling_dict.get(feature)
            logger.info(feature)
            logger.info(feature_params.get('type'))
            is_categorical.append(feature_params.get('type') == 'CATEGORY')
            is_text.append(feature_params.get('type') == 'TEXT')

        is_categorical = np.array(is_categorical)
        is_text = np.array(is_text)
        logger.info(is_text)
        #is_text = np.array([False] * len(selected_features))

        # Run the stress tests

        config_list = []

        if float(request.args.get('paramPS')) > 0:
            config_list.append(StressTestConfiguration(KnockOut(float(request.args.get('paramPS')))))

        if float(request.args.get('paramAA')) > 0:
            config_list.append(StressTestConfiguration(Adversarial(float(request.args.get('paramAA')))))

        if float(request.args.get('paramMV')) > 0:
            config_list.append(StressTestConfiguration(MissingValues(float(request.args.get('paramMV')))))

        if float(request.args.get('paramS')) > 0:
            config_list.append(StressTestConfiguration(Scaling(float(request.args.get('paramS')))))

        if float(request.args.get('paramT1')) > 0:
            config_list.append(StressTestConfiguration(ReplaceWord(float(request.args.get('paramT1')))))

        if float(request.args.get('paramT2')) > 0:
            config_list.append(StressTestConfiguration(Typos(float(request.args.get('paramT2')))))

        print('CONFIG LIST', config_list)
        stressor = StressTestGenerator(config_list, selected_features, is_categorical, is_text)
        perturbed_df = stressor.fit_transform(test_df, target_column=target)  # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_

        perturbed_df_with_prediction = model_accessor.predict(perturbed_df)
        reversed_target_mapping = {v: k for k, v in model_accessor.model_handler.get_target_map().items()}

        # Compute the performance drop metrics
        metrics_df = build_stress_metric(y_true=perturbed_df[target],
                                         y_pred=perturbed_df_with_prediction['prediction'],
                                         stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                         pos_label=reversed_target_mapping.get(1))

        name_mapping = {
            'ADVERSARIAL': 'Adversarial attack',
            'MISSING_VALUES': 'Missing values',
            'PRIOR_SHIFT': 'Prior shift',
            'SCALING': 'Scaling',
            'REPLACE_WORD': 'Replace Word',
            'TYPOS': 'Typos'
        }

        metrics_list = []
        for index, row in metrics_df.iterrows():
            dct = dict()
            dct['attack_type'] = name_mapping.get(row['_dku_stress_test_type'])
            dct['accuracy_drop'] = 100 * round(row['accuracy_drop'], 3)
            dct['robustness'] = 100 * round(row['robustness'], 3)
            metrics_list.append(dct)

        y_true = perturbed_df[target]

        original_target_value = list(model_accessor.model_handler.get_target_map().keys())
        y_true_class_confidence = perturbed_df_with_prediction[['proba_{}'.format(original_target_value[0]), 'proba_{}'.format(original_target_value[1])]].values
        y_true_idx = np.array([[True, False] if y == reversed_target_mapping.get(1) else [False, True] for y in y_true])
        y_true_class_confidence = y_true_class_confidence[y_true_idx]

        critical_samples_id_df = get_critical_samples(y_true_class_confidence=y_true_class_confidence,
                                                      stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                                      row_indicator=perturbed_df[DkuStressTestCenterConstants.DKU_ROW_ID])

        # TODO hot fix that should be done inside get_critical_samples
        critical_samples_id_df.reset_index(level=0, inplace=True)

        clean_df_with_id = perturbed_df.loc[perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE] == DkuStressTestCenterConstants.CLEAN].drop(DkuStressTestCenterConstants.STRESS_TEST_TYPE, axis=1)
        critical_samples_df = critical_samples_id_df.merge(clean_df_with_id, on=DkuStressTestCenterConstants.DKU_ROW_ID, how='left').drop(DkuStressTestCenterConstants.DKU_ROW_ID, axis=1)
        critical_samples_df['uncertainty'] = np.round(critical_samples_df['uncertainty'], 3)
        critical_samples_list = critical_samples_df.to_dict('records')

        data = {
            'metrics': metrics_list,
            'critical_samples': critical_samples_list
        }
        return simplejson.dumps(pretty_floats(data), ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500