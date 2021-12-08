import numpy as np
from math import ceil
import copy
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants
from collections import Counter

__all__ = ['OnlyOne', 'KnockOut', 'Rebalance']


class OnlyOne(Shift):
    """ Sample data to keep only one class.
    Args:
        keep_cl (int or str): class to keep
    Attributes:
        keep_cl (int or str): class to keep
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, keep_cl=0):
        super(OnlyOne, self).__init__()
        self.keep_cl = keep_cl
        self.name = 'oo_shift_%s' % keep_cl
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt = only_one_shift(Xt, yt, self.keep_cl)
        return Xt, yt


class KnockOut(Shift):
    """ Sample data to remove a portion of a given class.
    Args:
        cl (int or str): class to subsample
    Attributes:
        cl (int or str): class to subsample
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, cl=0, samples_fraction=1.0):
        super(KnockOut, self).__init__()
        self.cl = cl
        self.samples_fraction = samples_fraction
        self.name = 'ko_shift_%s_%.2f' % (cl, samples_fraction)
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt = knockout_shift(Xt, yt, self.cl, self.samples_fraction)
        return Xt, yt


class Rebalance(Shift):
    """ Sample data to match a given distribution of classes.
    Args:
        priors (dict): mapping of class -> desired frequency
    Attributes:
        priors (dict): mapping of class -> desired frequency
        name (str): name of the perturbation
        feature_type (int): identifier of the feature types for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, priors):
        super(Rebalance, self).__init__()
        if min(priors.values()) < 0:
            raise ValueError("Class frequencies cannot be negative")
        if sum(priors.values()) > 1:
            raise ValueError("The sum of the desired class frequencies exceeds 1")
        self.priors = priors
        self.name = 'rebalance_shift'
        for target_class, proba in priors.items():
            self.name += '_{}_{:.2f}'.format(target_class, proba)
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt, yt = rebalance_shift(X, y, self.priors)
        return Xt, yt


# Resample instances of all classes by given priors.
def rebalance_shift(x, y, priors):
    actual_class_counts = Counter(y)
    nr_samples = len(y)

    positive_priors = +Counter(priors)
    # Check current target distribution has all the classes in desired distribution
    if positive_priors.keys() > actual_class_counts.keys():
        raise ValueError(
            "One of the classes to resample is absent from the actual target distribution"
        )

    # If prior distribution is incomplete (sum < 1), we redistribute onto the unmapped classes
    # (i.e. classes in the actual distribution that are not in the desired prior distribution)
    nr_samples_to_redistribute = nr_samples * (1 - sum(positive_priors.values()))

    if nr_samples_to_redistribute == 0:
        redistribution_coef = 0
    else:
        # Redistribution can only be done if current distribution has some unmapped classes left
        unmapped_classes = actual_class_counts.keys() - priors.keys()
        if not unmapped_classes:
            raise ValueError("The desired prior distribution is incomplete")

        nr_samples_from_unmapped_classes = sum(actual_class_counts[target_class]
                                            for target_class in unmapped_classes)
        redistribution_coef = nr_samples_to_redistribute / nr_samples_from_unmapped_classes

    classes_to_resample = [
        target_class for target_class in actual_class_counts if priors.get(target_class) != 0
    ]
    class_to_initialize = classes_to_resample.pop()
    rebalanced_x_indices = np.random.choice(np.where(y==class_to_initialize)[0], y.shape)
    rebalanced_y = np.full(y.shape, class_to_initialize, dtype=y.dtype)
    offset = 0
    for target_class in classes_to_resample:
        desired_freq = priors.get(target_class)
        if desired_freq is None:
            desired_count = round(redistribution_coef * actual_class_counts[target_class])
        else:
            desired_count = round(desired_freq * nr_samples)
        if desired_count == 0:
            continue

        class_samples_indices = np.where(y==target_class)[0]
        desired_count = min(len(rebalanced_x_indices) - offset, desired_count)
        rebalanced_x_indices[offset : offset + desired_count] = np.random.choice(
            class_samples_indices, desired_count
        )
        rebalanced_y[offset : offset+desired_count] = target_class

        offset += desired_count
        if offset == len(rebalanced_x_indices):
            break
    return x[rebalanced_x_indices], rebalanced_y


# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    n_rows = x.shape[0]
    del_indices = np.where(y == cl)[0]
    until_index = ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)

    indices_cl = np.where(y == cl)[0]
    indices_not_cl = np.where(y != cl)[0]
    repeat_indices = np.random.choice(indices_not_cl, n_rows-len(indices_cl), replace=True)
    x_not_cl = x[repeat_indices, :]
    y_not_cl = y[repeat_indices]

    x = np.concatenate((x_not_cl, x[indices_cl, :]))
    y = np.concatenate((y_not_cl, y[indices_cl]))

    permuted_indices = np.random.permutation(n_rows)

    x = x[permuted_indices, :]
    y = y[permuted_indices]

    return x, y


# Remove all classes except for one via multiple knock-out.
def only_one_shift(x, y, keep_cl):
    labels = np.unique(y)
    for cl in labels:
        if cl != keep_cl:
            x, y = knockout_shift(x, y, cl, 1.0)

    return x, y
