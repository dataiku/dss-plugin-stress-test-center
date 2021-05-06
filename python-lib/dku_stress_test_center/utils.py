# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion
from drift_dac.prior_shift import KnockOut
from drift_dac.perturbation_shared_utils import Shift


def get_stress_test_name(shift: Shift):
    if isinstance(shift, MissingValues):
        return DkuStressTestCenterConstants.MISSING_VALUES
    elif isinstance(shift, Scaling):
        return DkuStressTestCenterConstants.SCALING
    elif isinstance(shift, Adversarial):
        return DkuStressTestCenterConstants.ADVERSARIAL
    elif isinstance(shift, KnockOut):
        return DkuStressTestCenterConstants.PRIOR_SHIFT
    elif isinstance(shift, ReplaceWord):
        return DkuStressTestCenterConstants.REPLACE_WORD
    elif isinstance(shift, Typos):
        return DkuStressTestCenterConstants.TYPOS
    elif isinstance(shift,WordDeletion):
        return DkuStressTestCenterConstants.WORD_DELETION
    else:
        raise NotImplementedError()


class DkuStressTestCenterConstants(object):
    CLEAN_DATASET_NUM_ROWS = 500
    CLEAN = 'CLEAN'
    MISSING_VALUES = 'MISSING_VALUES'
    SCALING = 'SCALING'
    ADVERSARIAL = 'ADVERSARIAL'
    PRIOR_SHIFT = 'PRIOR_SHIFT'
    REPLACE_WORD = 'REPLACE_WORD'
    TYPOS = 'TYPOS'
    WORD_DELETION = 'WORD_DELETION'

    PERTURBATION_BASED_STRESS_TYPES = [MISSING_VALUES, SCALING, ADVERSARIAL, REPLACE_WORD, TYPOS,WORD_DELETION]

    TARGET = 'target'

    CONFIDENCE = 'confidence'
    UNCERTAINTY = 'uncertainty'
    ACCURACY_DROP = 'accuracy_drop'
    F1_DROP = 'f1_drop'
    ROBUSTNESS = 'robustness'
    STRESS_TEST_TYPE = '_dku_stress_test_type'
    DKU_ROW_ID = '_dku_row_identifier_'

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

    REGRESSION_TYPE = 'REGRESSION'
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
