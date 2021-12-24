#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from dku_webapp import safe_str

from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler

def get_model_handler(model, version_id=None):
    try:
        params = model.get_predictor(version_id).params
        return PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder, params.model_folder)
    except Exception as e:
        from future.utils import raise_
        if "ordinal not in range(128)" in safe_str(e):
            raise_(Exception, "Model stress test only supports models built with Python 3. This one was built with Python 2.", sys.exc_info()[2])
        else:
            raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])
