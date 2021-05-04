# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import copy
from dku_model_parser.model_handler_utils import get_original_test_df
from dku_stress_test_center_utils import DkuStressTestCenterConstants, safe_str
from dataiku.doctor.posttraining.model_information_handler import PredictionModelInformationHandler
from drift_dac.perturbation_shared_utils import Shift
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='Stress Test Center Plugin | %(levelname)s - %(message)s')


'''

# example of how backend would call it
model_handler = DkuModelHandler(model, version_id)
model_handler.sample_data_from_dku_saved_model()

# from UI build a list of StressTestConfiguration list_of_stress_test_config

generator = StressTestGenerator(model_handler.clean_x, model_handler.clean_y, list_of_stress_test_config)
generator.run() # wait hours days, memory is filling up with datasets

evaluator = StressTestEvaluator(model_handler.clean_x, model_handler.clean_y, model_handler.model_predictor)
evaluator.run()

metrics_dict = evaluator.get_metrics()
critical_samples = evaluator.get_critical_samples()

'''


class DkuModelHandler(object):
    """
    DkuStressTestCenter
    """

    def __init__(self,
                 model,
                 version_id=None,
                 clean_dataset_size=DkuStressTestCenterConstants.CLEAN_DATASET_NUM_ROWS,
                 max_num_rows=DkuStressTestCenterConstants.MAX_NUM_ROWS,
                 random_state=65537):

        model_handler = self.get_model_handler(model, version_id)

        if model_handler is None:
            raise NotImplementedError('You need to define a model handler.')

        self._model_handler = model_handler
        self._target = model_handler.get_target_variable()
        # property
        self.model_predictor = model_handler.get_predictor()

        self._clean_dataset_size = clean_dataset_size
        self._max_num_rows = max_num_rows

        self._probability_threshold = self.model_predictor.params.model_perf.get('usedThreshold', None)
        self._feature_names = self.model_predictor.get_features()

        self._random_state = random_state

        # properties
        self.clean_x = None
        self.clean_y = None

    def get_model_handler(self, model, version_id=None):
        try:
            params = model.get_predictor(version_id).params
            return PredictionModelInformationHandler(params.split_desc, params.core_params, params.model_folder,
                                                     params.model_folder)
        except Exception as e:
            from future.utils import raise_
            if "ordinal not in range(128)" in safe_str(e):
                raise_(Exception, "Stress Test Center requires models built with python3. This one is on python2.",
                       sys.exc_info()[2])
            else:
                raise_(Exception, "Fail to load saved model: {}".format(e), sys.exc_info()[2])

    def get_original_test_df(self):
        try:
            return self._model_handler.get_test_df()[0]
        except Exception as e:
            logger.warning(
                'Cannot retrieve original test set: {}. The plugin will take the whole original dataset.'.format(e))
            return self._model_handler.get_full_df()[0]

    def sample_data_from_dku_saved_model(self):
        """  """
        np.random.seed(self._random_state)
        original_test_df = get_original_test_df(self._model_handler)[:self._max_num_rows]

        clean_df = original_test_df.sample(n=self._clean_dataset_size, random_state=self._random_state)

        self.clean_y = clean_df[self._target]
        self.clean_x = clean_df.drop(self._target, axis=1)



class StressTestConfiguration(object):
    def __init__(self,
                 shift_type: Shift,
                 list_of_features: list):
        self.shift = shift_type
        self.features = list_of_features
        # check valid configurations


class StressTestGenerator(object):
    def __init__(self,
                 clean_x: pd.DataFrame,
                 clean_y: pd.DataFrame,
                 config_list: list[StressTestConfiguration]):

        self.list_of_perturbed_datasets = []

    def run(self):
        for config in self.config_list:
            xt = copy.deepcopy(self.clean_x)
            yt = copy.deepcopy(self.clean_y)
            xt[:, config.features], yt = config.shift.transform(xt[:, config.features], yt)
            self.list_of_perturbed_datasets.append((xt, yt))


class StressTestEvaluator(object):
    def __init__(self,
                 clean_x,
                 clean_y,
                 model): # anything with predict/predict_proba

        # compute metrics on clean data here
        # store proba per sample
        pass

    def run(self, config_list, list_of_perturbed_datasets):

        for config, (perturbed_x, perturbed_y) in zip(config_list, list_of_perturbed_datasets):

            # batch evaluation
            # return average accuracy drop
            # return average f1 drop
            # return robustness metrics: 1-ASR or imbalanced accuracy for prior shift

            if config.shift.shifted_indices is not None: # it's perturbation based

                # critical samples evaluation
                # sort by std of uncertainty

    def get_critical_samples(self, top_k_samples=5):

        return None

    def get_metrics(self):
        # return dict with stress test config and its batch-level results

        return dict()





