import os
from collections.abc import Iterable
from copy import deepcopy
from typing import ItemsView

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pytsviz.viz import (
    plot_acf,
    plot_psd,
    plot_ts_analysis,
    plot_distribution,
    plot_gof,
    plot_ts,
    plot_seasonal_ts,
    plot_decomposed_ts,
    plot_forecast,
    plot_scatter_matrix,
    plot_scatter_fit,
    plot_inverse_arma_roots,
    plot_extended_scatter_matrix,
    plot_ts_overview,
)
from utils import root_path

data_path = os.path.join(root_path, "data", "crypto.csv")

df = pd.read_csv(data_path, index_col=0, parse_dates=True, dayfirst=True)

testing_dict = {
    plot_acf: {"df": df, "y_col": "LTC"},
    plot_psd: {"df": df, "y_col": "LTC"},
    plot_ts_analysis: {"df": df, "y_col": "LTC"},
    plot_distribution: {"df": df, "y_col": "LTC"},
    plot_gof: {"df": df, "y_col": "LTC", "y_hat_col": "LTC_fc"},
    plot_ts: {"df": df, "tf": "Box-Cox"},
    plot_seasonal_ts: {"df": df, "period": "year", "y_col": "LTC"},
    plot_decomposed_ts: {
        "df": df,
        "y_col": "LTC",
        "method": "STL",
        "period": 3,
    },
    plot_forecast: {
        "df": df,
        "y_col": "LTC",
        "y_hat_cols": ["LTC_fc"],
        "lower_col": "LTC_lb",
        "upper_col": "LTC_ub",
    },
    plot_scatter_matrix: {
        "df": df,
        "var1": "LTC",
        "var2": "BTC",
        "lags1": [5],
        "lags2": [5],
    },
    plot_scatter_fit: {"df": df, "x_col": "LTC", "y_col": "BTC"},
    plot_inverse_arma_roots: {
        "process": sm.tsa.ArmaProcess(
            np.r_[1, np.array([-0.75, 0.25])], np.r_[1, np.array([0.65, 0.35])]
        )
    },
    plot_extended_scatter_matrix: {"df": df},
    plot_ts_overview: {"df": df, "y_col": "LTC"},
}

func_names = [f.__name__ for f in testing_dict.keys()]


def base_typed(obj):
    """Recursive reflection method to convert any object property into a comparable form."""
    BASE_TYPES = [str, int, float, bool, type(None)]

    T = type(obj)
    from_numpy = T.__module__ == "numpy"

    if (
        T in BASE_TYPES
        or callable(obj)
        or (from_numpy and not isinstance(obj, Iterable))
    ):
        return obj

    if isinstance(obj, Iterable):
        base_items = [base_typed(item) for item in obj]
        return base_items if from_numpy else T(base_items)

    d = obj if T is dict else obj.__dict__

    return {k: base_typed(v) for k, v in d.items()}


def deep_equals(*args):
    return all(base_typed(args[0]) == base_typed(other) for other in args[1:])


@pytest.mark.parametrize("func_dict", testing_dict.items(), ids=func_names)
def test_args_are_unchanged(
    func_dict: ItemsView[str, dict],
):
    func, kwargs = func_dict
    processed_kwargs = deepcopy(kwargs)

    func(**processed_kwargs, show=False)

    for k in kwargs.keys():
        assert isinstance(kwargs[k], type(processed_kwargs[k]))
        if isinstance(kwargs[k], pd.DataFrame):
            pd.testing.assert_frame_equal(kwargs[k], processed_kwargs[k])
        elif isinstance(kwargs[k], sm.tsa.ArmaProcess):
            deep_equals(kwargs[k], processed_kwargs[k])
        else:
            assert kwargs[k] == processed_kwargs[k]
