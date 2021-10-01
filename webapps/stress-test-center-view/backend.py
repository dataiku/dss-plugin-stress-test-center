# -*- coding: utf-8 -*-
import numpy as np
import logging
import simplejson # might want to use flask.jsonify instead
import traceback
from flask import request, jsonify
import json

from dataiku import Model, api_client
from dataiku.customwebapp import get_webapp_config
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import build_stress_metric, get_critical_samples
from dku_stress_test_center.stress_test_center import DkuStressTestCenterConstants
from model_metadata import get_model_handler
from dku_webapp import convert_numpy_int64_to_int, pretty_floats

logger = logging.getLogger(__name__)

def undo_preproc_name(f):
    if ':' in f:
        return f.split(':')[1]
    else:
        return f

stressor = StressTestGenerator()

@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        logger.info('Retrieving model data...')
        fmi = get_webapp_config().get("trainedModelFullModelId")
        if fmi is None:
            model = Model(get_webapp_config()["modelId"])
            version_id = get_webapp_config().get("versionId")
            original_model_handler = get_model_handler(model, version_id)
        else:
            original_model_handler = PredictionModelInformationHandler.from_full_model_id(fmi)
        is_regression = 'REGRESSION' in original_model_handler.get_prediction_type()
        stressor.model_accessor = ModelAccessor(original_model_handler)
        return jsonify(
            target_classes=[] if is_regression else list(original_model_handler.get_target_map().keys()),
            columns={feature: preprocessing["type"] for (feature, preprocessing) in stressor.model_accessor.get_per_feature().items()}
        )
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route('/stress-tests-config', methods=["POST"])
def set_stress_tests_config():
    try:
        config = json.loads(request.data)
        stressor.set_config(config)
        return {"result": "ok"}
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route('/compute', methods=["GET"])
def compute():
    try:
        # Get test data
        model_accessor = stressor.model_accessor
        selected_features = set()
        feature_importance = model_accessor.get_feature_importance().index.tolist()

        for feature in feature_importance: # TODO: delete (use features per perturbation instead)
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
            is_categorical.append(feature_params.get('type') == 'CATEGORY')
            is_text.append(feature_params.get('type') == 'TEXT')

        is_categorical = np.array(is_categorical)
        is_text = np.array(is_text)

        reversed_target_mapping = {v: k for k, v in model_accessor.model_handler.get_target_map().items()}
        pos_label = reversed_target_mapping.get(1)

        # Run the stress tests
        stressor.selected_features = selected_features
        stressor.is_categorical = is_categorical
        stressor.is_text = is_text

        # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_
        perturbed_df = stressor.fit_transform()
        perturbed_df_with_prediction = model_accessor.predict(perturbed_df)


        # Compute the performance drop metrics
        target = model_accessor.get_target_variable()
        metrics_df = build_stress_metric(y_true=perturbed_df[target],
                                         y_pred=perturbed_df_with_prediction['prediction'],
                                         stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                         pos_label=pos_label)

        metrics_list = []
        for index, row in metrics_df.iterrows():
            dct = dict()
            dct['attack_type'] = row['_dku_stress_test_type']
            dct['accuracy_drop'] = round(100 * row['accuracy_drop'], 3)
            dct['robustness'] = round(100 * row['robustness'], 3)
            metrics_list.append(dct)


        y_true = perturbed_df[target]

        original_target_value = list(model_accessor.model_handler.get_target_map().keys())
        y_true_class_confidence = perturbed_df_with_prediction[
            ['proba_{}'.format(original_target_value[0]), 'proba_{}'.format(original_target_value[1])]
        ].values
        y_true_idx = np.array([
            [True, False] if y == reversed_target_mapping.get(1) else [False, True] for y in y_true
        ])
        y_true_class_confidence = y_true_class_confidence[y_true_idx]

        critical_samples_df, uncertainties = get_critical_samples(
            y_true_class_confidence=y_true_class_confidence,
            perturbed_df=perturbed_df
        )

        data = {
            'metrics': metrics_list,
            'critical_samples': critical_samples_df.to_dict('records'),
            'uncertainties': uncertainties
        }
        return simplejson.dumps(pretty_floats(data), ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500