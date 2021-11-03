# -*- coding: utf-8 -*-
import numpy as np
import logging
import simplejson # might want to use flask.jsonify instead
import traceback
from flask import request, jsonify
import json

from dataiku import Model
from dataiku.customwebapp import get_webapp_config
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestGenerator
from dku_stress_test_center.stress_test_center import get_critical_samples
from model_metadata import get_model_handler
from dku_webapp import convert_numpy_int64_to_int, pretty_floats

logger = logging.getLogger(__name__)

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
        stressor.model_accessor = ModelAccessor(original_model_handler)
        return jsonify(
            target_classes=stressor.model_accessor.get_target_classes(),
            features={
                feature: preprocessing["type"]
                for (feature, preprocessing) in stressor.model_accessor.get_per_feature().items()
                    if preprocessing["role"] == "INPUT"
            }
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
        stressor.fit_transform()

        # Compute the performance drop metrics
        target = model_accessor.get_target_variable()
        results = stressor.build_stress_metric()
 
        # Compute the critical samples
        #y_true = perturbed_df[target]

        #original_target_value = list(model_accessor.model_handler.get_target_map().keys())
        #y_true_class_confidence = perturbed_df_with_prediction[
        #    ['proba_{}'.format(original_target_value[0]), 'proba_{}'.format(original_target_value[1])]
        #].values
        #y_true_idx = np.array([
        #    [True, False] if y == reversed_target_mapping.get(1) else [False, True] for y in y_true
        #])
        #y_true_class_confidence = y_true_class_confidence[y_true_idx]

        #critical_samples_df, uncertainties = get_critical_samples(
        #    y_true_class_confidence=y_true_class_confidence,
        #    perturbed_df=perturbed_df
        #)

        return simplejson.dumps(pretty_floats(results), ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500
