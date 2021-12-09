# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos, WordDeletion
from drift_dac.prior_shift import Rebalance


class DkuStressTestCenterConstants(object):
    FEATURE_PERTURBATION = "FEATURE_PERTURBATION"
    SUBPOPULATION_SHIFT = "SUBPOPULATION_SHIFT"
    TARGET_SHIFT = "TARGET_SHIFT"

    TESTS = {
        MissingValues.__name__: (
            lambda **params: MissingValues(samples_fraction=params["samples_fraction"],
                                           features_fraction=1,
                                           value_to_put_in=None),
            FEATURE_PERTURBATION
        ),
        Scaling.__name__: (
            lambda **params: Scaling(samples_fraction=params["samples_fraction"],
                                     scaling_factor=params["scaling_factor"],
                                     features_fraction=1),
            FEATURE_PERTURBATION
        ),
        Adversarial.__name__: (Adversarial, FEATURE_PERTURBATION),
        Rebalance.__name__: (
            lambda **params: Rebalance(**params),
            TARGET_SHIFT
        ),
        ReplaceWord.__name__: (ReplaceWord, FEATURE_PERTURBATION),
        WordDeletion.__name__: (WordDeletion, FEATURE_PERTURBATION),
        Typos.__name__: (Typos, FEATURE_PERTURBATION)
    }

    NR_CRITICAL_SAMPLES = 5

    PREDICTION = "prediction"
    UNCORRUPTED = "_dku_stress_test_uncorrupted"

    MULTICLASS = "MULTICLASS"
    REGRESSION = "REGRESSION"
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
