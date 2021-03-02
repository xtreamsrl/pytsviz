import os
from copy import deepcopy
from typing import Callable, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pytsviz.global_vars import root_path
from pytsviz.viz import plot_acf, plotly_pacf, plot_psd, plot_ts_analysis, plot_distribution, plot_gof, \
    plot_ts, plot_seasonal_ts, plot_decomposed_ts, plot_forecast, plot_scatter_matrix, \
    plot_scatter_fit, plot_inverse_arma_roots, plot_extended_scatter_matrix, plot_ts_overview

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
    plot_acf: {
            "df": df,
            "y_col": "LTC"
    },
    plot_psd: {
            "df": df,
            "y_col": "LTC"
    },
    plot_ts_analysis: {
            "df": df,
            "y_col": "LTC"
    },
    plot_distribution: {
            "df": df,
            "y_col": "LTC"
    },
    plot_gof: {
            "df": df,
            "y_col": "LTC",
            "y_hat_col": "LTC_fc"
    },
    plot_ts: {
            "df": df,
            "tf": "moving_average"
    },
    plot_seasonal_ts: {
            "df": df,
            "period": "quarter",
            "y_col": "LTC"
    },
    plot_decomposed_ts: {
            "df": df,
            "method": "STL",
    },
    plot_forecast: {
            "df": df,
            "y_col": "LTC",
            "fc_cols": ["LTC_fc"],
            "lower_col": "LTC_lb",
            "upper_col": "LTC_ub"
    },
    plot_scatter_matrix: {
            "df": df,
            "var1": "LTC",
            "var2": "BTC",
            "lags1": [5],
            "lags2": [5]
    },
    plot_scatter_fit: {
            "df": df,
            "var1": "LTC",
            "var2": "BTC"
    },
    plot_inverse_arma_roots: {
            "process": sm.tsa.ArmaProcess(np.r_[1, np.array([-.75, .25])],  np.r_[1, np.array([.65, .35])])
    },
    plot_extended_scatter_matrix: {
            "df": df
    },
    plot_ts_overview: {
            "series": df["LTC"]
    }
}
