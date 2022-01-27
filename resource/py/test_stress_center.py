import pytest
import pandas as pd
import numpy as np

from dku_stress_test_center.stress_test_center import StressTestGenerator, FeaturePerturbationTest, SubpopulationShiftTest
from dku_stress_test_center.metrics import worst_group_performance, Metric

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

    def get_generator(weight="f2", regression=False):
        mocker.spy(generator, "_metric")
        generator._metric.name = "toto"
        generator._metric.compute.return_value = 42
        mocked_accessor.get_weight_variable.return_value = weight
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
    test.compute_specific_metrics.return_value = [{"name": "specific", "value": ".3"}]

    def get_mocked_test(regression=False):
        if regression:
            test.df_with_pred.drop(columns=["proba_A", "proba_B", "proba_C"], inplace=True)
        return test
    return get_mocked_test

def test_compute_test_metrics(mocker, stress_test_generator, stress_test):
    generator = stress_test_generator()
    test = stress_test()
    test.TEST_TYPE = "TARGET_SHIFT"
    result = generator.compute_test_metrics(test)

    corrup_y_true, corrup_y_pred, corrup_probas, corrup_weights = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas, clean_weights = generator._metric.compute.call_args_list[1][0]

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
    pd.testing.assert_series_equal(
        corrup_weights, pd.Series(["e", "e", "a", "a", "e"], name="f2", index=[1,2,5,0,3])
    )
    pd.testing.assert_series_equal(
        clean_weights, pd.Series(["a", "a", "a", "e", "a", "e"], name="f2")
    )
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
            {"name": "specific", "value": ".3"}
        ],
        "name": "test"
    }

    generator = stress_test_generator("f3", True)
    test = stress_test(True)
    test.TEST_TYPE = "FEATURE_PERTURBATION"
    result = generator.compute_test_metrics(test)

    corrup_y_true, corrup_y_pred, corrup_probas, corrup_weights = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas, clean_weights = generator._metric.compute.call_args_list[1][0]

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
    pd.testing.assert_series_equal(
        clean_weights, pd.Series([pd.np.nan, "b", "b", pd.np.nan, "c", "g"], name="f3")
    )
    pd.testing.assert_series_equal(
        corrup_weights, pd.Series(["b", "b", "b", "c", "g"], name="f3", index=[1,2,5,0,3])
    )
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
            {"name": "specific", "value": ".3"}
        ],
        "name": "test"
    }

    generator = stress_test_generator(None)
    test = stress_test()
    test.TEST_TYPE = "SUBPOPULATION_SHIFT"
    test.population = "f3"
    result = generator.compute_test_metrics(test)

    corrup_y_true, corrup_y_pred, corrup_probas, corrup_weights = generator._metric.compute.call_args_list[0][0]
    clean_y_true, clean_y_pred, clean_probas, clean_weights = generator._metric.compute.call_args_list[1][0]

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
    assert corrup_weights is None
    assert clean_weights is None
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
            {"name": "specific", "value": ".3"}
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
    pd.testing.assert_frame_equal(
        res["samples"], pd.DataFrame({"f1": [1,0,2], "f2": [3,4,2]}, index=[1,0,2])
    )
    predList = pd.DataFrame({"A": [1,0,2], "B": [50,40,7], "C": [1,1,1]}, index=[1,0,2])
    predList["C"] = predList["C"].astype(float)
    pd.testing.assert_frame_equal(
        res["predList"], predList
    )

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

def test_worst_subpop(mocker):
    mocked_metric = mocker.Mock()
    mocked_metric.compute.side_effect = lambda y_true, y_pred, probas, sample_weights: sum(y_true*y_pred + probas*(1 if sample_weights is None else sample_weights))
    mocked_metric.is_greater_better.return_value = True
    subpop = np.array(["a", "b", "b", "c", "c"])
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([50, 40, 30, 20, 10])
    probas = np.array([.5, .2, .3, .1, .2])
    sample_weights = np.array([4, 2, 1, 1, 4])
    res = worst_group_performance(mocked_metric, subpop, y_true, y_pred, probas, sample_weights)
    assert res == 52

    mocked_metric.is_greater_better.return_value = False
    res = worst_group_performance(mocked_metric, subpop, y_true, y_pred, probas, sample_weights)
    assert res == 170.7

    res = worst_group_performance(mocked_metric, subpop, y_true, y_pred, probas, None)
    assert res == 170.5
