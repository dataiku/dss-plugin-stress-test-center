import pytest
import pandas as pd
import numpy as np

from dku_stress_test_center.stress_test_center import StressTestGenerator, FeaturePerturbationTest, TargetShiftTest

@pytest.fixture
def stress_test_generator(mocker):
    mocked_accessor = mocker.Mock()
    mocked_accessor.get_target_variable.return_value = "target"
    mocked_accessor.get_target_map.return_value = {"A": 1, "B": 0, "C": 2}
    mocked_accessor.get_prediction_type.return_value = "REGRESSION"
    mocked_accessor.get_metric.return_value = "fake metric"

    generator = StressTestGenerator()
    generator._clean_df = pd.DataFrame({
        "f1": [1, 3, 2, 5, 10, 11],
        "f2": ["a", "a", "a", "e", "a", "e"],
        "f3": [np.nan, "b", "b", np.nan, "c", "g"],
        "target": ["C", "C", "B", "A", "A", "A"],
        "prediction": ["C", "A", "B", "A", "A", "B"],
        "proba_A": [.2, .2, .3, .3, .1, .9],
        "proba_B": [.1, .7, .5, .5, .5, .1],
        "proba_C": [.7, .1, .2, .2, .4, 0],
    })
    generator.model_accessor = mocked_accessor

    def get_generator(regression=False):
        if regression:
            generator._clean_df.drop(columns=["proba_A", "proba_B", "proba_C"], inplace=True)
        return generator
    return get_generator

@pytest.fixture
def stress_test(mocker):
    test = mocker.Mock(df_with_pred=pd.DataFrame({
        "f1": [0, 30, 12, -10, 0],
        "f2": ["e", "e", "a", "a", "e"],
        "f3": ["b", "b", "b", "c", "g"],
        "target": ["A", "B", "C", "C", "A"],
        "prediction": ["C", "C", "A", "C", "A"],
        "proba_A": [0, 0, 0, .1, .9],
        "proba_B": [.6, .5, .5, .4, 0],
        "proba_C": [.4, .5, .2, .5, .1],
    }, index=[1,2,5,0,3]))

    def get_mocked_test(regression=False):
        if regression:
            test.df_with_pred.drop(columns=["proba_A", "proba_B", "proba_C"], inplace=True)
        return test
    return get_mocked_test

def test_compute_test_metrics(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    generator.compute_test_metrics(test)

    metric, clean_y_true, corrup_y_true, clean_y_pred, corrup_y_pred, clean_probas, perturbed_probas = test.compute_metrics.call_args[0]
    assert metric == "fake metric"
    pd.testing.assert_series_equal(
        clean_y_true, pd.Series([2, 0, 1, 2, 1], name="target", index=[1,2,5,0,3])
    )
    pd.testing.assert_series_equal(
        corrup_y_true, pd.Series([1, 0, 2, 2, 1], name="target", index=[1,2,5,0,3])
        )
    pd.testing.assert_series_equal(
        clean_y_pred, pd.Series([1, 0, 0, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    pd.testing.assert_series_equal(
        corrup_y_pred, pd.Series([2, 2, 1, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    np.testing.assert_array_equal(clean_probas, np.array([
        [.2, .3, .9, .2, .3],
        [.7, .5, .1, .1, .5],
        [.1, .2, 0, .7, .2]
    ]).transpose())
    np.testing.assert_array_equal(perturbed_probas, np.array([
        [0, 0, 0, .1, .9],
        [.6, .5, .5, .4, 0],
        [.4, .5, .2, .5, .1]
    ]).transpose())

    generator = stress_test_generator(True)
    test = stress_test(True)
    generator.compute_test_metrics(test)

    metric, clean_y_true, corrup_y_true, clean_y_pred, corrup_y_pred, clean_probas, perturbed_probas = test.compute_metrics.call_args[0]
    assert metric == "fake metric"
    pd.testing.assert_series_equal(
        clean_y_true, pd.Series([2, 0, 1, 2, 1], name="target", index=[1,2,5,0,3])
    )
    pd.testing.assert_series_equal(
        corrup_y_true, pd.Series([1, 0, 2, 2, 1], name="target", index=[1,2,5,0,3])
        )
    pd.testing.assert_series_equal(
        clean_y_pred, pd.Series([1, 0, 0, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    pd.testing.assert_series_equal(
        corrup_y_pred, pd.Series([2, 2, 1, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    np.testing.assert_array_equal(clean_probas, np.empty((5,0)))
    np.testing.assert_array_equal(perturbed_probas, np.empty((5,0)))

def test__get_true_class_proba_columns(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    test.name = "test"
    other_test = mocker.Mock(df_with_pred=pd.DataFrame({
        "f1": [0, 30],
        "target": ["A", "B"],
        "prediction": ["C", "A"],
        "proba_A": [0, .9],
        "proba_B": [.6, 0],
        "proba_C": [.4, .5],
    }, index=[4, 0]))
    other_test.name ="other"

    generator.tests = { "FEATURE_PERTURBATION": [test, other_test]}
    df = generator._get_true_class_proba_columns("FEATURE_PERTURBATION")
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "_dku_stress_test_uncorrupted": [.7, .1, .5, .3, .1, .9],
        "test": [.5, .4, .5, .9, np.nan, 0],
        "other": [.5, np.nan, np.nan, np.nan, 0, np.nan]
    }))

def test__get_prediction_columns(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    test.name = "test"
    other_test = mocker.Mock(df_with_pred=pd.DataFrame({
        "f1": [0, 30],
        "target": ["A", "B"],
        "prediction": ["C", "A"],
        "proba_A": [0, .9],
        "proba_B": [.6, 0],
        "proba_C": [.4, .5],
    }, index=[4, 0]))
    other_test.name ="other"

    generator.tests = { "FEATURE_PERTURBATION": [test, other_test]}
    df = generator._get_prediction_columns("FEATURE_PERTURBATION")
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "_dku_stress_test_uncorrupted": ["C", "A", "B", "A", "A", "B"],
        "test": ["C", "C", "C", "A", np.nan, "A"],
        "other": ["A", np.nan, np.nan, np.nan, "C", np.nan]
    }))

def test_get_critical_samples(mocker, stress_test_generator):
    generator = stress_test_generator()
    generator.model_accessor.get_original_test_df.return_value = pd.DataFrame({
        "f1": range(5),
        "f2": range(4,-1, -1)
    })
    df = pd.DataFrame({
        "A": [0, 1, 2, 3],
        "B": [40, 50, 7, 6],
        "C": [1, 1, 1, np.nan]
    })
    mocker.patch.object(generator, "_get_prediction_columns", return_value=df)
    res = generator.get_critical_samples("does_not_matter")
    np.testing.assert_array_almost_equal(res["uncertainties"], [28.290163, 22.810816, 3.214550])
    np.testing.assert_array_almost_equal(res["means"], [17.333333, 13.666666, 3.333333])
    assert res["samples"] == [
        {"f1": 1,"f2": 3},
        {"f1": 0,"f2": 4},
        {"f1": 2,"f2": 2},
    ]
    assert res["predList"] == [
        {"A": 1,"B": 50, "C": 1},
        {"A": 0,"B": 40, "C": 1},
        {"A": 2,"B": 7, "C": 1},
    ]

def test_perturb_df(mocker):
    shift = mocker.Mock(feature_type=0)
    shift.transform.side_effect = lambda X: (-X*2, None)
    ft_corrupt = FeaturePerturbationTest(shift, features=["f1", "f3"])
    df = pd.DataFrame({
        "f1": range(5),
        "f2": np.arange(5)+1,
        "f3": np.arange(5)+3,
    })
    df = ft_corrupt.perturb_df(df)
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "f1": [0, -2, -4, -6, -8],
        "f2": np.arange(5)+1,
        "f3": [-6, -8, -10, -12, -14]
    }))

    shift = mocker.Mock(feature_type=0)
    shift.transform.side_effect = lambda X, Y: (-X*2, Y+"coucou")
    targ_shift = TargetShiftTest(shift, population="target")
    df = pd.DataFrame({
        "f1": range(5),
        "f2": np.arange(5)+1,
        "f3": np.arange(5)+3,
        "target": ["a", "b", "a", "b", "b"]
    })
    df = targ_shift.perturb_df(df)
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "f1": [0, -2, -4, -6, -8],
        "f2": [-2, -4, -6, -8, -10],
        "f3": [-6, -8, -10, -12, -14],
        "target": ["acoucou", "bcoucou", "acoucou", "bcoucou", "bcoucou"]
    }))
