# -*- coding: utf-8 -*-
import numpy as np
import logging
import simplejson # might want to use flask.jsonify instead
import traceback
from flask import request, jsonify
import json

from dataiku import Model
from dku_stress_test_center.utils import DkuStressTestCenterConstants
from dataiku.customwebapp import get_webapp_config
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

from dku_stress_test_center.metrics import Metric
from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestGenerator
from model_metadata import get_model_handler
from dku_webapp import convert_numpy_int64_to_int

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
            },
            metric={
                "initial": stressor.model_accessor.get_metric().initial,
                "actual": stressor.model_accessor.get_metric().actual,
                "greaterIsBetter": stressor.model_accessor.get_metric().is_greater_better()
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
        # Compute the performance drop metrics
        results = stressor.build_stress_metrics()
 
        # Compute the critical samples
        if DkuStressTestCenterConstants.FEATURE_PERTURBATION in results:
            results[DkuStressTestCenterConstants.FEATURE_PERTURBATION].update(
                critical_samples=stressor.get_critical_samples(
                    DkuStressTestCenterConstants.FEATURE_PERTURBATION
                )
            )

        return simplejson.dumps(results, ignore_nan=True, default=convert_numpy_int64_to_int)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500
