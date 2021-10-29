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

    PERTURBATION_BASED_STRESS_TYPES = [MISSING_VALUES, SCALING, ADVERSARIAL, REPLACE_WORD, TYPOS, WORD_DELETION]
    SUBPOP_SHIFT_BASED_STRESS_TYPES = [PRIOR_SHIFT]

    CONFIDENCE = 'confidence'
    UNCERTAINTY = 'uncertainty'
    ACCURACY_DROP = 'accuracy_drop'
    ROBUSTNESS = 'robustness'
    STRESS_TEST_TYPE = '_dku_stress_test_type'
    DKU_ROW_ID = '_dku_row_identifier_'

    MAX_NUM_ROW = 100000

    REGRESSION_TYPE = 'REGRESSION'
    CLASSIFICATION_TYPE = 'CLASSIFICATION'
    CLUSTERING_TYPE = 'CLUSTERING'
    DKU_CLASSIFICATION_TYPE = ['BINARY_CLASSIFICATION', 'MULTICLASS']
