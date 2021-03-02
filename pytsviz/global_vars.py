import math
from os.path import dirname

import numpy as np
import pandas as pd
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose

from pytsviz.utils import boxcox, yeojohnson, moving_average

root_path = dirname(dirname(__file__))

decomp_methods = {
    "STL": {STL: {}},
    "seasonal_additive": {seasonal_decompose: {"model": "additive"}},
    "seasonal_multiplicative": {
        seasonal_decompose: {"model": "multiplicative"}
    },
}

valid_components = [
    "level",
    "trend",
    "seasonal",
    "freq_seasonal",
    "cycle",
    "autoregressive",
    "resid",
]

valid_seasons = {
    "grouping": {
        "minute": lambda x: x.minute,
        "hour": lambda x: x.hour,
        "day": lambda x: x.isocalendar().day,
        "week": lambda x: x.isocalendar().week,
        "month": lambda x: x.month,
        "quarter": lambda x: (x.month - 1) // 3 + 1,
        "year": lambda x: x.isocalendar().year,
    },
    "granularity": {
        "minute": lambda x: x.second,
        "hour": lambda x: x.minute,
        "day": lambda x: x.hour,
        "week": lambda x: x.isocalendar().day,
        "month": lambda x: x.day,
        "quarter": lambda x: (x - pd.PeriodIndex(x, freq="Q").start_time).days
        + 1,
        "year": lambda x: x.dayofyear,
    },
}

transform_dict = {
    "Box-Cox": boxcox,
    "Yeo-Johnson": yeojohnson,
    "log": np.vectorize(math.log),
    "moving_average": moving_average,
}
