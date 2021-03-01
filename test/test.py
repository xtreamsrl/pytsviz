import os
from copy import deepcopy
from typing import Callable, Tuple
import pandas as pd

from pytsviz.global_vars import root_path
from pytsviz.viz import plotly_acf, plotly_pacf, plotly_psd, plotly_tsdisplay, plot_distribution_histogram, plot_gof, \
    time_series_plot, seasonal_time_series_plot, decomposed_time_series_plot, forecast_plot, vars_scatterplot, \
    scatterplot, inverse_arma_roots_plot, composite_matrix_scatterplot, composite_summary_plot

data_path = os.path.join(root_path, "data", "crypto.csv")

df = pd.read_csv(data_path)
print(df)


def test_args_are_unchanged(
        func: Callable,
        args: Tuple,
        kwargs: dict
):
    input_args = deepcopy(args)
    input_kwargs = deepcopy(kwargs)

    func(args, kwargs)

    for i in range(len(args)):
        assert args[i] == input_args[i]

    for i in range(len(kwargs)):
        assert kwargs[i] == input_kwargs[i]


testing_dict = {
    plotly_acf: {
        "args": (df["LTC"], 50),
        "kwargs": {}
    },
    plotly_pacf: {
        "args": (df["LTC"], 50),
        "kwargs": {}
    },
    plotly_psd: {
        "args": (series, nlags),
        "kwargs": {}
    },
    plotly_tsdisplay: {
        "args": (series, nlags),
        "kwargs": {}
    },
    plot_distribution_histogram: {
        "args": (series, nlags),
        "kwargs": {}
    },
    plot_gof: {
        "args": (series, nlags),
        "kwargs": {}
    },
    time_series_plot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    seasonal_time_series_plot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    decomposed_time_series_plot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    forecast_plot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    vars_scatterplot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    scatterplot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    inverse_arma_roots_plot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    composite_matrix_scatterplot: {
        "args": (series, nlags),
        "kwargs": {}
    },
    composite_summary_plot: {
        "args": (series, nlags),
        "kwargs": {}
    }
}
