import numpy as np
import copy

from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants

__all__ = ['MissingValues', 'Scaling', 'sample_random_indices']

def sample_random_indices(total_size, fraction):
    if fraction == 1:
        return np.arange(total_size)
    nr_samples_to_pick = int(np.ceil(fraction * total_size))
    indices = np.random.choice(total_size, size=nr_samples_to_pick, replace=False)
    indices.sort()
    return indices

class MissingValues(Shift):
    """ Insert missing values into a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        value_to_put_in (any, default np.nan): desired representation of the missing value
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        value_to_put_in (float): desired representation of the missing value
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, samples_fraction=1.0, features_fraction=1.0, value_to_put_in=np.nan):
        super(MissingValues, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.value_to_put_in = value_to_put_in

        self.name = 'missing_value_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        if X.dtype <= int and (self.value_to_put_in is None or isinstance(self.value_to_put_in, float)):
            Xt = X.astype(float)
        else:
            Xt = copy.deepcopy(X)

        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        self.shifted_features = sample_random_indices(Xt.shape[1], self.features_fraction)

        Xt[np.ix_(self.shifted_indices, self.shifted_features)] = self.value_to_put_in

        return Xt, yt


class Scaling(Shift):
    """ Scale a portion of samples and features by a value, that can be either randomly selected
    or set.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        scaling_factor (float, default None): value to use for the scaling. If None, a random value is picked
        amongst [10, 100, 1000].
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, samples_fraction=1.0, features_fraction=1.0, scaling_factor=None):
        super(Scaling, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self._scaling_factor = scaling_factor
        self.name = 'scaling_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    @property
    def scaling_factor(self):
        if self._scaling_factor is None:
            return np.random.choice([10, 100, 1000])
        return self._scaling_factor

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        if X.dtype <= int and isinstance(self.scaling_factor, float):
            Xt = X.astype(float)
        else:
            Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        self.shifted_features = sample_random_indices(Xt.shape[1], self.features_fraction)

        Xt[np.ix_(self.shifted_indices, self.shifted_features)] *= self.scaling_factor

        return Xt, yt
