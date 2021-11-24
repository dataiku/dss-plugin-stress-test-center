import numpy as np
from sklearn.metrics import accuracy_score

def get_performance_metric(metric_name: str):
    return accuracy_score # TODO

def greater_perf_is_better(metric_name: str):
    return True # TODO

def worst_group_performance(metric_name: str, y_true: np.array, y_pred: np.array,
                            subpopulation: np.array):
    performances = []
    subpopulation_values = np.unique(subpopulation)
    for subpop in subpopulation_values:
        subpop_mask = subpopulation == subpop
        performance = get_performance_metric(metric_name)(y_true[subpop_mask], y_pred[subpop_mask])
        performances.append(performance)
    return min(performances) if greater_perf_is_better(metric_name) else max(performances)

def stress_resilience(clean_y_pred: np.array, perturbed_y_pred: np.array):
    # TODO: make it work for regression as well
    return (clean_y_pred == perturbed_y_pred).sum() / len(clean_y_pred)

def performance_variation(metric_name: str, clean_y_true: np.array, clean_y_pred: np.array,
                          perturbed_y_true: np.array, perturbed_y_pred: np.array):
    greater_is_better = greater_perf_is_better(metric_name)
    performance_metric = get_performance_metric(metric_name)

    clean_performance_metric = performance_metric(clean_y_true, clean_y_pred)
    stressed_performance_metric = performance_metric(perturbed_y_true, perturbed_y_pred)

    delta = stressed_performance_metric - clean_performance_metric
    return delta if greater_is_better else - delta # TODO might be trickier than this
