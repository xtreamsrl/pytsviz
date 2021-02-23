"""
The *utils* module contains utilies not strictly related to visualization which we often use (eg harmonics computation).
"""

import datetime as dt
import math
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.structural import UnobservedComponents


def harmonics(dates, period, n, epoch=datetime(1900, 1, 1)):
    """
    Computes harmonics for the given dates. Each harmonic is made of a couple of sinusoidal and cosinusoidal waves
    with frequency i/period, i = 1...n. The argument of the functions is the number of hours from the starting epoch.

    :param dates: a pandas series of dates
    :type dates: :py:class:`pd.Series <pandas:pandas.Series>` of :py:class:`python:datetime.datetime`
    :param period: the base period of the harmonics
    :type period: `int`
    :param n: the number of harmonics to include
    :type n: `int`
    :param epoch: the epoch used to compute the argument of the sin
    :type epoch: :py:class:`python:datetime.datetime`
    :return: a Pandas DataFrame with dates as index and harmonics as columns
    :rtype: :py:class:`pandas:pandas.DataFrame`
    """
    d = pd.DataFrame(index=dates)
    hours = (dates - epoch) / pd.Timedelta(hours=1)

    for i in range(1, n + 1):
        d[f"Sin_{round(period)}_{i}"] = np.sin(2 * i * np.pi * hours / period)
        d[f"Cos_{round(period)}_{i}"] = np.cos(2 * i * np.pi * hours / period)

    return d


def datetimediv(dividend, delta):
    seconds = int((dividend - dt.datetime.min).total_seconds())
    remainder = dt.timedelta(
        seconds=seconds % delta.total_seconds(),
        microseconds=dividend.microsecond,
    )
    quotient = dividend - remainder
    return quotient, remainder


def boxcox(x):
    return stats.boxcox(x)[0]


def yeojohnson(x):
    return stats.yeojohnson(x)[0]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


transform_dict = {
    "Box-Cox": {
        boxcox: {
            }
    },
    "Yeo-Johnson": {
        yeojohnson: {
        }
    },
    "log": {
        np.vectorize(math.log): {
        }
    },
    "moving_average": {
        moving_average: {
        }
    }
}

decompose_dict = {
    "STL": {
        STL: {
        }
    },
    "seasonal_additive": {
        seasonal_decompose: {
            "model": "additive"
        }
    },
    "seasonal_multiplicative": {
        seasonal_decompose: {
            "model": "multiplicative"
        }
    },
    "harmonic": {
        UnobservedComponents: {
            "level": "fixed intercept",
            "freq_seasonal": [{'period': 50}],

        }
    }
}

valid_components = [
    "level",
    "trend",
    "seasonal",
    "freq_seasonal",
    "cycle",
    "autoregressive",
    "resid"
]

valid_seasons = {
    "grouping": {
        "minute": lambda x: x.minute,
        "hour": lambda x: x.hour,
        "day": lambda x: x.isocalendar().day,
        "week": lambda x: x.isocalendar().week,
        "month": lambda x: x.month,
        "quarter": lambda x: (x.month - 1) // 3 + 1,
        "year": lambda x: x.isocalendar().year
    },
    "granularity": {
        "minute": lambda x: x.second,
        "hour": lambda x: x.minute,
        "day": lambda x: x.hour,
        "week": lambda x: x.isocalendar().day,
        "month": lambda x: x.day,
        "quarter": lambda x: (x - pd.PeriodIndex(x, freq='Q').start_time).days + 1,
        "year": lambda x: x.dayofyear
    }
}


def set_time_index(df, time_col):
    if time_col is not None:
        df.set_index(time_col, inplace=True)


def get_components(result):
    components = {}
    for c in valid_components:
        if c in dir(result):
            components[c] = getattr(result, c)
    return components
