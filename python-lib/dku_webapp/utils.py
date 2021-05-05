import dataiku
import pandas as pd
import numpy as np
import logging
from dku_webapp.webapp_constants import DkuWebappConstants

logger = logging.getLogger(__name__)

def convert_numpy_int64_to_int(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


class PrettyFloat(float):
    def __repr__(self):
        return '%.15g' % self


def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(pretty_floats, obj))
    return obj

