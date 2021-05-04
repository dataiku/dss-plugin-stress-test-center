# -*- coding: utf-8 -*-
import sys


def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    if isinstance(val, unicode):
        return val.encode("utf-8")
    return str(val)


class DkuStressTestCenterConstants(object):
    CLEAN_DATASET_NUM_ROWS = 500
    MAX_NUM_ROWS = 100000