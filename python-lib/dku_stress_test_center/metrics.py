import numpy as np
from math import sqrt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,\
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from dataiku.doctor.utils.metrics import rmsle_score, mroc_auc_score, mean_absolute_percentage_error, log_loss
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

    def __init__(self, name, config=None, pred_type=None):
        self.config = config
        self.name = name
        self.pred_type = pred_type

    def is_greater_better(self):
        return self.name in self.GREATER_IS_BETTER

    def compute(self, y_true, y_pred, probas, sample_weight):
        if self.pred_type == DkuStressTestCenterConstants.MULTICLASS:
            extra_params = {
                "average": 'macro',
                "pos_label": None
            }
        else:
            extra_params = {}

        perf_metric = {
            self.F1: lambda y_true, y_pred, probas: f1_score(y_true, y_pred, sample_weight=sample_weight, **extra_params),
            self.ACCURACY: lambda y_true, y_pred, probas: accuracy_score(y_true, y_pred, sample_weight=sample_weight),
            self.RECALL: lambda y_true, y_pred, probas:\
                recall_score(y_true, y_pred, sample_weight=sample_weight, **extra_params),
            self.PRECISION: lambda y_true, y_pred, probas:\
                precision_score(y_true, y_pred, sample_weight=sample_weight, **extra_params),
            self.COST_MATRIX: lambda y_true, y_pred, probas:\
                make_cost_matrix_score(self.config)(y_true, y_pred, sample_weight=sample_weight) / float(len(y_true)),
            self.CUMULATIVE_LIFT: lambda y_true, y_pred, probas:\
                make_lift_score(self.config)(y_true, probas, sample_weight=sample_weight),
            self.LOG_LOSS: lambda y_true, y_pred, probas: log_loss(y_true, probas, sample_weight=sample_weight),
            self.ROC_AUC: lambda y_true, y_pred, probas: mroc_auc_score(y_true, probas, sample_weight=sample_weight),
            self.EVS: lambda y_true, y_pred, probas: explained_variance_score(y_true, y_pred, sample_weight=sample_weight),
            self.MAPE: lambda y_true, y_pred, probas:\
                mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight),
            self.MAE: lambda y_true, y_pred, probas: mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
            self.MSE: lambda y_true, y_pred, probas: mean_squared_error(y_true, y_pred, sample_weight=sample_weight),
            self.RMSE: lambda y_true, y_pred, probas: sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
            self.RMSLE: lambda y_true, y_pred, probas: rmsle_score(y_true, y_pred, sample_weight=sample_weight),
            self.R2: lambda y_true, y_pred, probas: r2_score(y_true, y_pred, sample_weight=sample_weight),
        }.get(self.name)

        if perf_metric is None:
            raise ValueError("Unknown evaluation metric: {}".format(self.name))
        return perf_metric(y_true, y_pred, probas)


def worst_group_performance(metric, subpopulation, y_true,
                            y_pred, probas, sample_weights):
    performances = []
    subpopulation_values = np.unique(subpopulation)
    for subpop in subpopulation_values:
        subpop_mask = subpopulation == subpop
        subpop_weights =  None if sample_weights is None else sample_weights[subpop_mask]
        performance = metric.compute(
            y_true[subpop_mask], y_pred[subpop_mask], probas[subpop_mask], subpop_weights
        )
        performances.append(performance)
    return min(performances) if metric.is_greater_better() else max(performances)


def corruption_resilience_classification(clean_y_pred, perturbed_y_pred):
    return (clean_y_pred == perturbed_y_pred).sum() / float(len(clean_y_pred))


def corruption_resilience_regression(clean_y_pred, perturbed_y_pred,
                                 clean_y_true):
    clean_abs_error = np.abs(clean_y_pred - clean_y_true)
    perturbed_abs_error = np.abs(perturbed_y_pred - clean_y_true)
    return np.count_nonzero(perturbed_abs_error <= clean_abs_error) / float(len(clean_y_true))
