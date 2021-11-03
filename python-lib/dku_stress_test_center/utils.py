# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion
from drift_dac.prior_shift import KnockOut


class DkuStressTestCenterConstants(object):
    CLEAN = 'CLEAN'
    TESTS = {
        MissingValues.__name__: MissingValues,
        Scaling.__name__: Scaling,
        Adversarial.__name__: Adversarial,
        KnockOut.__name__: KnockOut,
        ReplaceWord.__name__: ReplaceWord,
        WordDeletion.__name__: WordDeletion,
        Typos.__name__: Typos
    }

    FEATURE_PERTURBATIONS = [MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion]
    SAMPLING_PERTURBATIONS = [KnockOut]

    CONFIDENCE = 'confidence'
    UNCERTAINTY = 'uncertainty'
    ACCURACY_DROP = 'accuracy_drop'
    ROBUSTNESS = 'robustness'
    STRESS_TEST_TYPE = '_dku_stress_test_type'
    DKU_ROW_ID = '_dku_row_identifier_'

    REGRESSION_TYPE = 'REGRESSION'
    CLASSIFICATION_TYPE = 'CLASSIFICATION'
    CLUSTERING_TYPE = 'CLUSTERING'
    DKU_CLASSIFICATION_TYPE = ['BINARY_CLASSIFICATION', 'MULTICLASS']
