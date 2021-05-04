# -*- coding: utf-8 -*-
from dku_stress_test_center.model_accessor import ModelAccessor
from dku_stress_test_center.stress_test_center import StressTestConfiguration, StressTestGenerator
from dku_stress_test_center.stress_test_center import build_stress_metric, get_critical_samples
from dku_stress_test_center.stress_test_center import DkuStressTestCenterConstants
from drift_dac.covariate_shift import MissingValues, Scaling, Adversarial
from model_metadata import get_model_handler
from drift_dac.prior_shift import KnockOut
import dataiku


def dummy_backend(model_id, version_id):
    # Access the model
    model = dataiku.Model(model_id)
    model_handler = get_model_handler(model=model, version_id=version_id)
    model_accessor = ModelAccessor(model_handler)

    # Get test data
    test_df = model_accessor.get_original_test_df()
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
    test_df = test_df[selected_features]

    # Run the stress tests
    config_list = [StressTestConfiguration(KnockOut),
                   StressTestConfiguration(MissingValues),
                   StressTestConfiguration(Scaling),
                   StressTestConfiguration(Adversarial)]

    stressor = StressTestGenerator(config_list)
    perturbed_df = stressor.fit_transform(
        test_df)  # perturbed_df is a dataset of schema feat_1 | feat_2 | ... | _STRESS_TEST_TYPE | _DKU_ID_

    perturbed_df_with_prediction = model_accessor.predict(perturbed_df)

    # Compute the performance drop metrics
    metrics_df = build_stress_metric(y_true=perturbed_df_with_prediction['target'],
                                     y_pred=perturbed_df_with_prediction['prediction'],
                                     stress_test_indicator=perturbed_df_with_prediction[
                                         DkuStressTestCenterConstants.STRESS_TEST_TYPE])

    critical_samples_df = get_critical_samples(y_pred=perturbed_df_with_prediction['target'],
                                               y_proba=perturbed_df_with_prediction[['proba_<=50K', 'proba_>50K']],
                                               stress_test_indicator=perturbed_df_with_prediction[
                                                   DkuStressTestCenterConstants.STRESS_TEST_TYPE],
                                               row_indicator=perturbed_df_with_prediction[
                                                   DkuStressTestCenterConstants.DKU_ROW_ID])
