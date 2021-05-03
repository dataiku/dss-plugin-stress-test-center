from drift_dac.perturbation_shared_utils import Shift
from sklearn.metrics import accuracy_score, f1_score
import copy

# the backend has
# n_bootstrap # n. of random selections of the dataset from the whole test set
# list of the shift types (each with its severity, its n_runs, and its list of features to perturb if applicable)

# the backend calls
# StressDatasetSampler to get a clean dataset to perturb (we compare to this one? not the original dataset?)
# For each clean dataset and each user-selected shift/features it calls StressTest (which in turns run the perturbations
# and use a StressImpactEvaluator to compute the metrics)
# Get metrics and critical samples from StressTest


class StressDatasetSampler(object):
    def __init__(self, dss_model_handler, n_samples=1000):
        self.dss_model_handler = dss_model_handler
        self.n_samples = n_samples

    def run(self):
        # get X_test, y_test as the original test set of the input DSS model

        # subsample

        X_clean = X_test_subsampled
        y_clean = y_test_subsampled

        return X_clean, y_clean


class StressTest(object):
    def __init__(self, dss_model_handler, stress=Shift, features=None, n_runs=1):
        self.dss_model_handler = dss_model_handler
        self.stress = stress  # severity included in Shift type
        self.features = features
        self.n_runs = n_runs  # n. of random perturbations of the selected dataset (to compute critical samples uncertainty)
        self.evaluator = None

    def run(self, X_clean, y_clean):

        self.evaluator = StressImpactEvaluator(self.dss_model_handler, X_clean, y_clean)

        # run perturbation on subsample n_runs times
        X_perturbed = copy.deepcopy(X_clean)
        (X_perturbed[:, self.features], y_perturbed) = self.stress.transform(X_clean[:, self.features], y_clean)

        self.evaluator.evaluate(X_perturbed, y_perturbed)

    def get_critical_samples(self, top_k_samples=5):
        # sort by std of uncertainty
        pass

    def get_robustness_metrics(self):
        # return average accuracy drop
        # return average f1 drop
        # return robustness metrics: 1-ASR or imbalanced accuracy
        pass



class StressImpactEvaluator(object):
    def __init__(self, dss_model_handler, X_clean, y_clean, sample_level_metrics=False):
        self.dss_model_handler = dss_model_handler

        self.sample_level_metrics = sample_level_metrics # where to get this info: is it a perturbation-based or selection-based corruption

        # compute metrics on clean data at init
        self.y_prob_clean = self.dss_model_handler.predict_proba(X_clean)
        self.y_pred_clean = self.dss_model_handler.predict(X_clean)

        self.metrics_clean = {
            'accuracy': accuracy_score(y_clean, self.y_pred_clean),
            'f1': f1_score(y_clean, self.y_pred_clean)
        }

        self.metrics_perturbed = None

    def evaluate(self, X_perturbed, y_perturbed):
        # for several evaluate, accumulate results

        self.y_prob_perturbed = self.dss_model_handler.predict_proba(X_perturbed)
        self.y_pred_perturbed = self.dss_model_handler.predict(X_perturbed)

        if self.metrics_perturbed is None:
            self.metrics_perturbed = {
                'accuracy': [accuracy_score(y_perturbed, self.y_pred_perturbed)],
                'f1': [f1_score(y_perturbed, self.y_pred_perturbed)]
            }
        else:
            self.metrics_perturbed['accuracy'].append(accuracy_score(y_perturbed, self.y_pred_perturbed))
            self.metrics_perturbed['f1'].append(f1_score(y_perturbed, self.y_pred_perturbed))

        # Robustness

        # Compute and accumulate drop

        # IF perturbation-based
        if self.sample_level_metrics:
            pass
            # compute and accumulate 1 - Attack Success Rate

            # store n_runs per-sample confidences

            # compute and accumulate per-sample confidence change

        # also need a way to compute extra metric for Prior shift: imbalanced accuracy

    def get_critical_samples(self, top_k_samples=5):
        # sort by std of uncertainty
        pass

    def get_robustness_metrics(self):
        # return average accuracy drop
        # return average f1 drop
        # return robustness metrics: 1-ASR or imbalanced accuracy
        pass






