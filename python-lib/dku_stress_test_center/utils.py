# -*- coding: utf-8 -*-
import sys
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial
from drift_dac.prior_shift import KnockOut
from drift_dac.perturbation_shared_utils import Shift


def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)


def get_stress_test_name(shift: Shift):
    if shift == MissingValues:
        return DkuStressTestCenterConstants.MISSING_VALUES
    elif shift == Scaling:
        return DkuStressTestCenterConstants.SCALING
    elif shift == Adversarial:
        return DkuStressTestCenterConstants.ADVERSARIAL
    elif shift == KnockOut:
        return DkuStressTestCenterConstants.PRIOR_SHIFT
    #elif shift == TextAttack:
    #    return DkuStressTestCenterConstants.TEXTATTACK
    else:
        raise NotImplementedError()


class DkuStressTestCenterConstants(object):
    CLEAN_DATASET_NUM_ROWS = 500
    MISSING_VALUES = 'MISSING_VALUES'
    SCALING = 'SCALING'
    ADVERSARIAL = 'ADVERSARIAL'
    PRIOR_SHIFT = 'PRIOR_SHIFT'

    TIMESTAMP = 'timestamp'
    MODEL_ID = 'model_id'
    VERSION_ID = 'version_id'
    TRAIN_DATE = 'train_date'

    MOST_IMPORTANT_FEATURES = 'most_important_features_in_deployed_model'
    MOST_IMPORTANT_FEATURES_DEFINTIION = 'Most important features in the deployed model, with their % of importance (max 20 features).'
    FEATURE_IMPORTANCE = 'feature_importance'

    MIN_NUM_ROWS = 500
    MAX_NUM_ROW = 100000
    CUMULATIVE_PERCENTAGE_THRESHOLD = 90
    PREDICTION_TEST_SIZE = 100000
    SURROGATE_TARGET = "_dku_predicted_label_"

    REGRRSSION_TYPE = 'REGRESSION'
    CLASSIFICATION_TYPE = 'CLASSIFICATION'
    CLUSTERING_TYPE = 'CLUSTERING'
    DKU_CLASSIFICATION_TYPE = ['BINARY_CLASSIFICATION', 'MULTICLASS']

    FEAT_IMP_CUMULATIVE_PERCENTAGE_THRESHOLD = 95

    FEATURE = 'feature'
    IMPORTANCE = 'importance'
    CUMULATIVE_IMPORTANCE = 'cumulative_importance'
    RANK = 'rank'
    CLASS = 'class'
    PERCENTAGE = 'percentage'
    ORIGINAL_DATASET = 'original_dataset'
    NEW_DATASET = 'new_dataset'
