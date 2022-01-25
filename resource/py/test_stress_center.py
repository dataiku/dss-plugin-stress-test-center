import pytest
import pandas as pd
import numpy as np
from dku_stress_test_center.stress_test_center import StressTestGenerator, FeaturePerturbationTest, SubpopulationShiftTest

@pytest.fixture
def stress_test_generator(mocker):
    mocked_accessor = mocker.Mock()
    mocked_accessor.get_target_variable.return_value = "target"
    mocked_accessor.get_target_map.return_value = {"A": 1, "B": 0, "C": 2}
    mocked_accessor.get_prediction_type.return_value = "REGRESSION"

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
        mocker.spy(generator, "_metric")
        generator._metric.name = "toto"
        generator._metric.compute.return_value = 42
        if regression:
            generator._clean_df = generator._clean_df.drop(columns=["proba_A", "proba_B", "proba_C"])
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
    test.name = "test"

    def get_mocked_test(regression=False):
        if regression:
            test.df_with_pred.drop(columns=["proba_A", "proba_B", "proba_C"], inplace=True)
        return test
    return get_mocked_test

def test_compute_test_metrics(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    test.TEST_TYPE = "TARGET_SHIFT"
    result = generator.compute_test_metrics(test, generator._clean_df)

    corrup_y_true, corrup_y_pred, corrup_probas = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas = generator._metric.compute.call_args_list[1][0]

    pd.testing.assert_series_equal(
        clean_y_true, pd.Series([2, 2, 0, 1, 1, 1], name="target")
    )
    pd.testing.assert_series_equal(
        corrup_y_true, pd.Series([1, 0, 2, 2, 1], name="target", index=[1,2,5,0,3])
        )
    pd.testing.assert_series_equal(
        clean_y_pred, pd.Series([2, 1, 0, 1, 1, 0], name="prediction")
    )
    pd.testing.assert_series_equal(
        corrup_y_pred, pd.Series([2, 2, 1, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    np.testing.assert_array_equal(clean_probas, np.array([
        [.2, .2, .3, .3, .1, .9],
        [.1, .7, .5, .5, .5, .1],
        [.7, .1, .2, .2, .4, 0]
    ]).transpose())
    np.testing.assert_array_equal(corrup_probas, np.array([
        [0, 0, 0, .1, .9],
        [.6, .5, .5, .4, 0],
        [.4, .5, .2, .5, .1]
    ]).transpose())
    assert result == {
        "metrics": [
            {
                "name": "perf_before",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_after",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_var",
                "base_metric": "toto",
                "value": 0
            }
        ],
        "name": "test"
    }

    generator = stress_test_generator(True)
    test = stress_test(True)
    test.TEST_TYPE = "FEATURE_PERTURBATION"
    mocker.patch("dku_stress_test_center.stress_test_center.corruption_resilience_regression", return_value=.2)
    mocker.patch("dku_stress_test_center.stress_test_center.corruption_resilience_classification", return_value=.8)
    result = generator.compute_test_metrics(test, generator._clean_df)

    corrup_y_true, corrup_y_pred, corrup_probas = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas = generator._metric.compute.call_args_list[1][0]

    pd.testing.assert_series_equal(
        clean_y_true, pd.Series([2, 2, 0, 1, 1, 1], name="target")
    )
    pd.testing.assert_series_equal(
        corrup_y_true, pd.Series([1, 0, 2, 2, 1], name="target", index=[1,2,5,0,3])
        )
    pd.testing.assert_series_equal(
        clean_y_pred, pd.Series([2, 1, 0, 1, 1, 0], name="prediction")
    )
    pd.testing.assert_series_equal(
        corrup_y_pred, pd.Series([2, 2, 1, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    np.testing.assert_array_equal(clean_probas, np.empty((6,0)))
    np.testing.assert_array_equal(corrup_probas, np.empty((5,0)))
    assert result == {
        "metrics": [
            {
                "name": "perf_before",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_after",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_var",
                "base_metric": "toto",
                "value": 0
            },
            {
                "name": "corruption_resilience",
                "value": .2
            }
        ],
        "name": "test"
    }

    generator = stress_test_generator()
    test = stress_test()
    test.TEST_TYPE = "SUBPOPULATION_SHIFT"
    test.population = "f3"
    mocker.patch("dku_stress_test_center.stress_test_center.worst_group_performance", return_value=.42)
    result = generator.compute_test_metrics(test, generator._clean_df)

    corrup_y_true, corrup_y_pred, corrup_probas = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas = generator._metric.compute.call_args_list[1][0]

    pd.testing.assert_series_equal(
        clean_y_true, pd.Series([2, 2, 0, 1, 1, 1], name="target")
    )
    pd.testing.assert_series_equal(
        corrup_y_true, pd.Series([1, 0, 2, 2, 1], name="target", index=[1,2,5,0,3])
        )
    pd.testing.assert_series_equal(
        clean_y_pred, pd.Series([2, 1, 0, 1, 1, 0], name="prediction")
    )
    pd.testing.assert_series_equal(
        corrup_y_pred, pd.Series([2, 2, 1, 2, 1], name="prediction", index=[1,2,5,0,3])
    )
    np.testing.assert_array_equal(clean_probas, np.empty((6,0)))
    np.testing.assert_array_equal(corrup_probas, np.empty((5,0)))
    assert result == {
        "metrics": [
            {
                "name": "perf_before",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_after",
                "base_metric": "toto",
                "value": 42
            },
            {
                "name": "perf_var",
                "base_metric": "toto",
                "value": 0
            },
            {
                "name": "worst_subpop_perf",
                "base_metric": "toto",
                "value": .42
            }
        ],
        "name": "test"
    }

def test__get_true_class_proba_columns(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    other_test = mocker.Mock(df_with_pred=pd.DataFrame({
        "f1": [0, 30],
        "target": ["A", "B"],
        "prediction": ["C", "A"],
        "proba_A": [0, .9],
        "proba_B": [.6, 0],
        "proba_C": [.4, .5],
    }, index=[4, 0]))
    other_test.name ="other"

    generator._tests = { "FEATURE_PERTURBATION": [test, other_test]}
    df = generator._get_true_class_proba_columns("FEATURE_PERTURBATION")
    pd.testing.assert_frame_equal(df, pd.DataFrame({
        "_dku_stress_test_uncorrupted": [.7, .1, .5, .3, .1, .9],
        "test": [.5, .4, .5, .9, np.nan, 0],
        "other": [.5, np.nan, np.nan, np.nan, 0, np.nan]
    }))

def test__get_prediction_columns(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    other_test = mocker.Mock(df_with_pred=pd.DataFrame({
        "f1": [0, 30],
        "target": ["A", "B"],
        "prediction": ["C", "A"],
        "proba_A": [0, .9],
        "proba_B": [.6, 0],
        "proba_C": [.4, .5],
    }, index=[4, 0]))
    other_test.name ="other"

    generator._tests = { "FEATURE_PERTURBATION": [test, other_test]}
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
    ft_corrupt = FeaturePerturbationTest(
        "MissingValues", params={"samples_fraction":.5}, selected_features=["f1", "f3"]
    )
    ft_corrupt.shift = mocker.Mock(feature_type=0)
    ft_corrupt.shift.transform.side_effect = lambda X: (-X*2, None)

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

    targ_shift = SubpopulationShiftTest(
        "MissingValues", params={"samples_fraction":.5}, population="target"
    )
    targ_shift.shift = mocker.Mock(feature_type=0)
    targ_shift.shift.transform.side_effect = lambda X, Y: (-X*2, Y+"coucou")
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
