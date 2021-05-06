# -*- coding: utf-8 -*-
from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import build_stress_metric, get_critical_samples
from dku_stress_test_center.stress_test_center import DkuStressTestCenterConstants
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial, ReplaceWord, Typos
from model_metadata import get_model_handler
from drift_dac.prior_shift import KnockOut
import dataiku
import numpy as np

def dummy_backend(model_id, version_id):
    # Access the model
    model = dataiku.Model(model_id)
    model_handler = get_model_handler(model=model, version_id=version_id)
    model_accessor = ModelAccessor(model_handler)

    # Get test data
    test_df = model_accessor.get_original_test_df()
    target = model_accessor.get_target_variable()
    # selected_features = ...  # either top 10 feats or later given by the user.

    def undo_preproc_name(f):
        if ':' in f:
            return f.split(':')[1]
        else:
            return f

    selected_features = set()
    feature_importance = model_accessor.get_feature_importance().index.tolist()

    feat_id = 0
    while len(selected_features) < 10:
        selected_features.add(undo_preproc_name(feature_importance[feat_id]))
        feat_id += 1

    selected_features = list(selected_features)

    # no we cannot filter before, we need all features to end up in perturbed_df to do the prediction afterwords
    #test_df = test_df[selected_features + [target]]

    # Run the stress tests
    config_list = [StressTestConfiguration(KnockOut()),
                   StressTestConfiguration(MissingValues()),
                   StressTestConfiguration(Scaling()),
                   StressTestConfiguration(Adversarial()),
                   StressTestConfiguration(ReplaceWord()),
                   StressTestConfiguration(Typos())]

    stressor = StressTestGenerator(config_list, selected_features, is_categorical, is_text)
    perturbed_df = stressor.fit_transform(
        test_df)  # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_

    perturbed_df_with_prediction = model_accessor.predict(perturbed_df)

    # Compute the performance drop metrics
    metrics_df = build_stress_metric(y_true=perturbed_df[target],
                                     y_pred=perturbed_df_with_prediction['prediction'],
                                     stress_test_indicator=perturbed_df[DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                     pos_label='>50K')

    y_true = perturbed_df[target]
    y_true_class_confidence = perturbed_df_with_prediction[['proba_<=50K', 'proba_>50K']].values
    y_true_idx = np.array([[True, False] if y == '>=50K' else [False, True] for y in y_true])
    y_true_class_confidence = y_true_class_confidence[y_true_idx]

    critical_samples_df = get_critical_samples(y_true_class_confidence=y_true_class_confidence,
                                               stress_test_indicator=perturbed_df[
                                                   DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                               row_indicator=perturbed_df[DkuStressTestCenterConstants.DKU_ROW_ID])
