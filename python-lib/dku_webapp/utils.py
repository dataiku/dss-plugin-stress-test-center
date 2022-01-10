import numpy as np
import sys
import json

class DKUJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)
