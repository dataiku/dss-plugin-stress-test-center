# -*- coding: utf-8 -*-
from drift_dac.covariate_shift import MissingValues, Scaling
from drift_dac.prior_shift import Rebalance


class DkuStressTestCenterConstants(object):
    FEATURE_PERTURBATION = "FEATURE_PERTURBATION"
    SUBPOPULATION_SHIFT = "SUBPOPULATION_SHIFT"
    TARGET_SHIFT = "TARGET_SHIFT"

    TESTS = {
        "MissingValues": lambda **params: MissingValues(
            samples_fraction=params["samples_fraction"], features_fraction=1, value_to_put_in=None
        ),
        "Scaling": lambda **params: Scaling(
            samples_fraction=params["samples_fraction"], scaling_factor=params["scaling_factor"],
            features_fraction=1
        ),
        "RebalanceTarget": lambda **params: Rebalance(**params),
        "RebalanceFeature": lambda **params: Rebalance(**params)
    }

    NR_CRITICAL_SAMPLES = 5

    PREDICTION = "prediction"
    UNCORRUPTED = "_dku_stress_test_uncorrupted"

    MULTICLASS = "MULTICLASS"
    REGRESSION = "REGRESSION"
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
