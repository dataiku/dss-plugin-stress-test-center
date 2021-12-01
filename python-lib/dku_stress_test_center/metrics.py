import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, log_loss,\
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from dataiku.doctor.utils.metrics import rmsle_score, mroc_auc_score, mean_absolute_percentage_error
from dataiku.doctor.prediction.common import make_lift_score, make_cost_matrix_score

from dku_stress_test_center.utils import DkuStressTestCenterConstants

class Metric(object):
    F1="F1"
    ACCURACY="ACCURACY"
    RECALL="RECALL"
    PRECISION="PRECISION"
    COST_MATRIX="COST_MATRIX"
    CUMULATIVE_LIFT="CUMULATIVE_LIFT"
    LOG_LOSS="LOG_LOSS"
    ROC_AUC="ROC_AUC"
    EVS="EVS"
    MAPE="MAPE"
    MAE="MAE"
    MSE="MSE"
    RMSE="RMSE"
    RMSLE="RMSLE"
    R2="R2"
    CUSTOM="CUSTOM"
    GREATER_IS_BETTER={ACCURACY, PRECISION, RECALL, F1, COST_MATRIX, ROC_AUC, CUMULATIVE_LIFT, EVS, R2}

    def __init__(self, config: dict, pred_type: str):
        self.pred_type = pred_type
        self.config = config

    @property
    def name(self):
        metric_name = self.config["evaluationMetric"]
        if metric_name == self.CUSTOM:
            if self.pred_type == DkuStressTestCenterConstants.REGRESSION:
                return self.R2
            return self.ACCURACY
        return metric_name

    def is_greater_better(self):
        return self.name in self.GREATER_IS_BETTER

    def get_performance_metric(self, y_true: np.array, y_pred: np.array, probas: np.array):
        if self.pred_type == DkuStressTestCenterConstants.MULTICLASS:
            extra_params = {
                "average": 'macro',
                "pos_label": None
            }
        else:
            extra_params = {}

        perf_metric = {
            self.F1: lambda y_true, y_pred, probas: f1_score(y_true, y_pred, **extra_params),
            self.ACCURACY: lambda y_true, y_pred, probas: accuracy_score(y_true, y_pred),
            self.RECALL: lambda y_true, y_pred, probas:\
                recall_score(y_true, y_pred, **extra_params),
            self.PRECISION: lambda y_true, y_pred, probas:\
                precision_score(y_true, y_pred, **extra_params),
            self.COST_MATRIX: lambda y_true, y_pred, probas:\
                make_cost_matrix_score(metrics)(y_true, y_pred) / len(y_true),
            self.CUMULATIVE_LIFT: lambda y_true, y_pred, probas:\
                make_lift_score(metrics)(y_true, probas),
            self.LOG_LOSS: lambda y_true, y_pred, probas: log_loss(y_true, probas),
            self.ROC_AUC: lambda y_true, y_pred, probas: mroc_auc_score(y_true, probas),
            self.EVS: lambda y_true, y_pred, probas: explained_variance_score(y_true, y_pred),
            self.MAPE: lambda y_true, y_pred, probas:\
                mean_absolute_percentage_error(y_true, y_pred),
            self.MAE: lambda y_true, y_pred, probas: mean_absolute_error(y_true, y_pred),
            self.MSE: lambda y_true, y_pred, probas: mean_squared_error(y_true, y_pred),
            self.RMSE: lambda y_true, y_pred, probas: sqrt(mean_squared_error(y_true, y_pred)),
            self.RMSLE: lambda y_true, y_pred, probas: rmsle_score(y_true, y_pred),
            self.R2: lambda y_true, y_pred, probas: r2_score(y_true, y_pred),
        }.get(self.name)

        if perf_metric is None:
            raise ValueError("Unknown training metric: {}".format(self.name))
        return perf_metric(y_true, y_pred, probas)


def worst_group_performance(metric: Metric, subpopulation: np.array, y_true: np.array,
                            y_pred: np.array, probas: np.array):
    performances = []
    subpopulation_values = np.unique(subpopulation)
    for subpop in subpopulation_values:
        subpop_mask = subpopulation == subpop
        performance = metric.get_performance_metric(y_true[subpop_mask], y_pred[subpop_mask], probas)
        performances.append(performance)
    return min(performances) if metric.is_greater_better() else max(performances)


def stress_resilience_classification(clean_y_pred: np.array, perturbed_y_pred: np.array):
    return (clean_y_pred == perturbed_y_pred).sum() / len(clean_y_pred)


def stress_resilience_regression(clean_y_pred: np.array, perturbed_y_pred: np.array,
                                 clean_y_true: np.array):
    clean_abs_error = np.abs(clean_y_pred - clean_y_true)
    perturbed_abs_error = np.abs(perturbed_y_pred - clean_y_true)
    return np.count_nonzero(perturbed_abs_error <= clean_abs_error) / len(clean_y_true)


def performance_variation(metric: Metric, clean_y_true: np.array, perturbed_y_true: np.array,
                          clean_y_pred: np.array, perturbed_y_pred: np.array,
                          clean_probas: np.array, perturbed_probas: np.array):
    clean_performance_metric = metric.get_performance_metric(
        clean_y_true, clean_y_pred, clean_probas
    )
    stressed_performance_metric = metric.get_performance_metric(
        perturbed_y_true, perturbed_y_pred, perturbed_probas
    )

    delta = stressed_performance_metric - clean_performance_metric
    return delta if metric.is_greater_better() else - delta
