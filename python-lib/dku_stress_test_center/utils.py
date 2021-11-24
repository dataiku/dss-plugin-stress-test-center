# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion
from drift_dac.prior_shift import Rebalance


class DkuStressTestCenterConstants(object):
    FEATURE_PERTURBATION = "FEATURE_PERTURBATION"
    SUBPOPULATION_SHIFT = "SUBPOPULATION_SHIFT"

    TESTS = {
        MissingValues.__name__: (MissingValues, FEATURE_PERTURBATION),
        Scaling.__name__: (Scaling, FEATURE_PERTURBATION),
        Adversarial.__name__: (Adversarial, FEATURE_PERTURBATION),
        Rebalance.__name__: (
            lambda **params: Rebalance({params["cl"]: params["samples_fraction"]}),
            SUBPOPULATION_SHIFT
        ),
        ReplaceWord.__name__: (ReplaceWord, FEATURE_PERTURBATION),
        WordDeletion.__name__: (WordDeletion, FEATURE_PERTURBATION),
        Typos.__name__: (Typos, FEATURE_PERTURBATION)
    }

    NR_CRITICAL_SAMPLES = 5

    PREDICTION = 'prediction'

    REGRESSION_TYPE = 'REGRESSION'
    CLASSIFICATION_TYPE = 'CLASSIFICATION'
    CLUSTERING_TYPE = 'CLUSTERING'
    DKU_CLASSIFICATION_TYPE = ['BINARY_CLASSIFICATION', 'MULTICLASS']
