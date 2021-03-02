"""
The *utils* module contains utilies not strictly related to visualization which we often use (eg harmonics computation).
"""

import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd
from colour import Color
from scipy import stats
from pytsviz import global_vars


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


def boxcox(x):
    return stats.boxcox(x)[0]


def yeojohnson(x):
    return stats.yeojohnson(x)[0]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def set_time_index(df, time_col):
    if time_col is not None:
        df.set_index(time_col, inplace=True)


def get_components(result):
    components = {}
    for c in global_vars.valid_components:
        if c in dir(result):
            components[c] = getattr(result, c)
    return components


def make_granular_colorscale(col1, col2, n):
    return [c.hex for c in Color(col1).range_to(col2, n)]


def apply_grad_color_to_traces(fig, col1, col2):
    n_traces = len(fig['data'])
    custom_colorscale = make_granular_colorscale(col1, col2, n_traces)
    for i in range(n_traces):
        fig['data'][i]['line']['color'] = custom_colorscale[i]
