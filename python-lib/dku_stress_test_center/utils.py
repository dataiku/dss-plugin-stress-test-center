# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion
from drift_dac.prior_shift import KnockOut


class DkuStressTestCenterConstants(object):
    CLEAN = 'CLEAN'

    FEATURE_PERTURBATION = "FEATURE_PERTURBATION"
    SUBPOPULATION_SHIFT = "SUBPOPULATION_SHIFT"

    TESTS = {
        MissingValues.__name__: (MissingValues, FEATURE_PERTURBATION),
        Scaling.__name__: (Scaling, FEATURE_PERTURBATION),
        Adversarial.__name__: (Adversarial, FEATURE_PERTURBATION),
        KnockOut.__name__: (KnockOut, SUBPOPULATION_SHIFT),
        ReplaceWord.__name__: (ReplaceWord, FEATURE_PERTURBATION),
        WordDeletion.__name__: (WordDeletion, FEATURE_PERTURBATION),
        Typos.__name__: (Typos, FEATURE_PERTURBATION)
    }

    NR_CRITICAL_SAMPLES = 5

    PREDICTION = 'prediction'
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
