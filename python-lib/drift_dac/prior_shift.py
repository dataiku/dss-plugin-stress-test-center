import numpy as np
from math import ceil
import copy
from dku_webapp import safe_str
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants
from collections import Counter

__all__ = ['Rebalance']

# This class is used both for rebalancing the distribution of target classes and feature categories
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
            raise ValueError("Frequencies cannot be negative")
        if sum(priors.values()) > 1:
            raise ValueError("The sum of the desired frequencies exceeds 1")
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

    positive_priors = Counter() + Counter(priors)
    # Check current target distribution has all the classes in desired distribution
    if set(positive_priors.keys()) - set(actual_class_counts.keys()):
        raise ValueError(
            "One of the modalities to resample is absent from the actual distribution"
        )

    # If prior distribution is incomplete (sum < 1), we redistribute onto the unmapped classes
    # (i.e. classes in the actual distribution that are not in the desired prior distribution)
    nr_samples_to_redistribute = nr_samples * (1 - sum(positive_priors.values()))

    if nr_samples_to_redistribute == 0:
        redistribution_coef = 0
    else:
        # Redistribution can only be done if current distribution has some unmapped classes left
        unmapped_classes = set(actual_class_counts.keys()) - set(priors.keys())
        if not unmapped_classes:
            raise ValueError("The desired distribution is incomplete")

        nr_samples_from_unmapped_classes = sum(actual_class_counts[target_class]
                                            for target_class in unmapped_classes)
        redistribution_coef = nr_samples_to_redistribute / float(nr_samples_from_unmapped_classes)

    class_with_non_null_priors = [
        target_class for target_class in actual_class_counts if priors.get(target_class, 0) != 0
    ]
    class_with_null_priors = [
        target_class for target_class in actual_class_counts if priors.get(target_class) is None
    ]
    if class_with_non_null_priors:
        class_to_initialize = class_with_non_null_priors[-1]
        offset = round(priors[class_to_initialize] * nr_samples)
    else:
        class_to_initialize = class_with_null_priors[-1]
        offset = round(redistribution_coef * actual_class_counts[class_to_initialize])
    if type(class_to_initialize) is float and np.isnan(class_to_initialize):
        subpopulation_indices = np.where(y.apply(safe_str) == safe_str(class_to_initialize))[0]
    else:
        subpopulation_indices = np.where(y == class_to_initialize)[0]
    rebalanced_x_indices = np.random.choice(subpopulation_indices, y.size)
    rebalanced_y = np.full(y.shape, class_to_initialize, dtype=y.dtype)

    classes_to_resample = class_with_non_null_priors + class_with_null_priors
    for target_class in classes_to_resample:
        if target_class == class_to_initialize:
            continue
        desired_freq = priors.get(target_class)
        if desired_freq is None:
            desired_count = round(redistribution_coef * actual_class_counts[target_class])
        else:
            desired_count = round(desired_freq * nr_samples)
        if desired_count == 0:
            continue
        desired_count = min(len(rebalanced_x_indices) - offset, desired_count)
        if type(target_class) is float and np.isnan(target_class):
            subpopulation_indices = np.where(y.astype(safe_str) == safe_str(target_class))[0]
        else:
            subpopulation_indices = np.where(y == target_class)[0]
        desired_count, offset = int(desired_count), int(offset)
        rebalanced_x_indices[offset : desired_count + offset] = np.random.choice(
            subpopulation_indices, desired_count
        )
        rebalanced_y[offset : offset + desired_count] = target_class

        offset += desired_count
        if offset == len(rebalanced_x_indices):
            break
    return x[rebalanced_x_indices], rebalanced_y
