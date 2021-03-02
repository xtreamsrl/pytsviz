import os
from copy import deepcopy
from typing import Callable, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pytsviz.global_vars import root_path
from pytsviz.viz import plotly_acf, plotly_pacf, plotly_psd, plotly_tsdisplay, plot_distribution_histogram, plot_gof, \
    time_series_plot, seasonal_time_series_plot, decomposed_time_series_plot, forecast_plot, vars_scatterplot, \
    scatterplot, inverse_arma_roots_plot, composite_matrix_scatterplot, composite_summary_plot

data_path = os.path.join(root_path, "data", "crypto.csv")

df = pd.read_csv(data_path, index_col=0, parse_dates=True, dayfirst=True)


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
            "series": df["LTC"],
            "nlags": 50
    },
    plotly_pacf: {
            "series": df["LTC"],
            "nlags": 50
    },
    plotly_psd: {
            "series": df["LTC"]
    },
    plotly_tsdisplay: {
            "series": df["LTC"]
    },
    plot_distribution_histogram: {
            "series": df["LTC"]
    },
    plot_gof: {
            "df": df,
            "y_col": "LTC",
            "y_hat_col": "LTC_fc"
    },
    time_series_plot: {
            "df": df,
            "tf": "moving_average"
    },
    seasonal_time_series_plot: {
            "df": df,
            "period": "quarter",
            "y_col": "LTC"
    },
    decomposed_time_series_plot: {
            "df": df,
            "method": "STL",
    },
    forecast_plot: {
            "df": df,
            "y_col": "LTC",
            "fc_cols": ["LTC_fc"],
            "lower_col": "LTC_lb",
            "upper_col": "LTC_ub"
    },
    vars_scatterplot: {
            "df": df,
            "var1": "LTC",
            "var2": "BTC",
            "lags1": [5],
            "lags2": [5]
    },
    scatterplot: {
            "df": df,
            "var1": "LTC",
            "var2": "BTC"
    },
    inverse_arma_roots_plot: {
            "process": sm.tsa.ArmaProcess(np.r_[1, np.array([-.75, .25])],  np.r_[1, np.array([.65, .35])])
    },
    composite_matrix_scatterplot: {
            "df": df
    },
    composite_summary_plot: {
            "series": df["LTC"]
    }
}
