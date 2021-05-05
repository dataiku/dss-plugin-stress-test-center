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

