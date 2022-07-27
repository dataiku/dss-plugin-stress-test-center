import numpy as np
import logging
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
from dku_webapp import DKUJSONEncoder

app.json_encoder = DKUJSONEncoder

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
            fmi = "S-{project_key}-{model_id}-{version_id}".format(
                project_key=model.projet_key, model_id=model.get_id(), version_id=version_id
            )
        original_model_handler = PredictionModelInformationHandler.from_full_model_id(fmi)
        stressor.model_accessor = ModelAccessor(original_model_handler)

        return jsonify(
            target_classes=stressor.model_accessor.get_target_classes(),
            pred_type=stressor.model_accessor.get_prediction_type(),
            features={
                feature: preprocessing["type"]
                for (feature, preprocessing) in stressor.model_accessor.get_per_feature().items()
                    if preprocessing["role"] == "INPUT"
            },
            metric=stressor.model_accessor.get_evaluation_metric()
        )
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route('/stress-tests-config', methods=["POST"])
def set_stress_tests_config():
    try:
        config = json.loads(request.data)
        stressor.set_config(config)
        return jsonify(result="ok")
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500

@app.route('/compute', methods=["GET"])
def compute():
    try:
        # Compute the performance drop metrics
        results = stressor.build_results()
 
        # Compute the critical samples
        if DkuStressTestCenterConstants.FEATURE_PERTURBATION in results:
            results[DkuStressTestCenterConstants.FEATURE_PERTURBATION].update(
                critical_samples=stressor.get_critical_samples(
                    DkuStressTestCenterConstants.FEATURE_PERTURBATION
                )
            )

        return jsonify(results)
    except:
        logger.error("When trying to call compute endpoint: {}.".format(traceback.format_exc()))
        return "{}. Check backend log for more details.".format(traceback.format_exc()), 500

@app.route('/<string:feature>/categories', methods=["GET"])
def get_feature_categories(feature):
    try:
        categories = stressor.model_accessor.get_categories(feature)
        return jsonify(categories)
    except:
        logger.error(traceback.format_exc())
        return traceback.format_exc(), 500
