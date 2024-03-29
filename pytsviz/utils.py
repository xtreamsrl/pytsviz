"""
The *utils* module contains utilities not strictly related to visualization which we often use (eg harmonics computation).
"""

import math
from datetime import datetime
from os.path import dirname

import numpy as np
import pandas as pd
from colour import Color
from scipy import stats
from statsmodels.tsa._stl import STL
from statsmodels.tsa.seasonal import seasonal_decompose


def harmonics(
    dates: pd.DataFrame,
    period: int,
    n: int,
    epoch: datetime = datetime(1900, 1, 1),
) -> pd.DataFrame:
    """
    Computes harmonics for the given dates. Each harmonic is made of a couple of sinusoidal and cosinusoidal waves
    with frequency i/period, i = 1...n. The argument of the functions is the number of hours from the starting epoch.

    :param dates: A pandas series of dates.
    :param period: The base period of the harmonics.
    :param n: The number of harmonics to include.
    :param epoch: The epoch used to compute the argument of the sin.
    :return: A Pandas DataFrame with dates as index and harmonics as columns.
    """
    d = pd.DataFrame(index=dates)
    hours = (dates - epoch) / pd.Timedelta(hours=1)

    for i in range(1, n + 1):
        d[f"Sin_{round(period)}_{i}"] = np.sin(2 * i * np.pi * hours / period)
        d[f"Cos_{round(period)}_{i}"] = np.cos(2 * i * np.pi * hours / period)

    return d


def boxcox(x):
    return stats.boxcox(x)[0]


def yeojohnson(x):
    return stats.yeojohnson(x)[0]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "same") / w


def set_time_index(df, time_col):
    if time_col is not None:
        df.set_index(time_col, inplace=True)


def get_components(result):
    components = {}
    for c in valid_components:
        if c in dir(result):
            components[c] = getattr(result, c)
    return components


def make_granular_colorscale(col1, col2, n):
    return [c.hex for c in Color(col1).range_to(col2, n)]


def apply_grad_color_to_traces(fig, col1, col2):
    n_traces = len(fig["data"])
    custom_colorscale = make_granular_colorscale(col1, col2, n_traces)
    for i in range(n_traces):
        fig["data"][i]["line"]["color"] = custom_colorscale[i]


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
