"""The *viz* module contains functions to visualize most of the key aspects of a univariate time series such as
(partial) correlograms, periodograms, line plots, ... """

from copy import deepcopy
from itertools import product
from typing import (
    List,
    Callable,
    Iterable,
    Tuple,
    Any,
    Union,
    Literal,
    Optional,
)

import plotly
import plotly.express as px
import plotly.graph_objs as go
from numpy.core import linspace
from plotly.basedatatypes import BaseFigure
from plotly.subplots import make_subplots
from scipy.signal import periodogram
from scipy.stats import pearsonr
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf, pacf

from .utils import *

colorway = plotly.colors.qualitative.Dark24
seq_colorscale = plotly.colors.sequential.PuBuGn
seq_colorscale_bounds = [
    "#FFF7FB",
    "#014636",
]  # extremes of "PuBuGn", for building arbitrarily granular color grads
div_colorscale = plotly.colors.diverging.Fall

template = dict(
    layout=go.Layout(
        font=dict(family="Rockwell"),
        xaxis=dict(showgrid=False, zeroline=False, automargin=True),
        yaxis=dict(zeroline=False),
        title_font=dict(size=24),
        autosize=True,
        height=800,
        margin=dict(l=75, r=75, b=75, t=75, pad=3),
        legend_title=dict(text=""),
        colorscale=dict(
            diverging=div_colorscale,
            sequential=seq_colorscale,
            sequentialminus=seq_colorscale[::-1],
        ),
        colorway=colorway,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
)


def _plot_plotly(
    df: pd.DataFrame,
    kind: str,
    x_title: str = None,
    y_title: str = None,
    x_type: Literal[
        "linear", "log", "date", "category", "multicategory"
    ] = None,
    y_type: Literal[
        "linear", "log", "date", "category", "multicategory"
    ] = None,
    **kwargs: Any,
) -> BaseFigure:
    """
    Wrapper for pandas plot function, with plotly backend and the application of the default styling template.

    :param df: Dataframe to use as the data source for the plot.
    :param kind: The kind of plot to produce.
    :param x_title: x axis title.
    :param y_title: y axis title.
    :param x_type: If provided, enforces the x axis data type. Can be one of 'linear', 'log', 'date', 'category',
        'multicategory'.
    :param y_type: If provided, enforces the y axis data type. Can be one of 'linear', 'log', 'date', 'category',
        'multicategory'.
    :param kwargs: Keyword arguments for pandas.plot.
    :return: Plotly figure.
    """
    with pd.option_context("plotting.backend", "plotly"):
        fig = df.plot(kind=kind, template=template, **kwargs)
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_type=x_type,
        yaxis_type=y_type,
    )
    return fig


def plot_acf(
    df: pd.DataFrame,
    y_col: str = None,
    time_col: str = None,
    partial: bool = False,
    n_lags: int = None,
    title: str = None,
    show_threshold: bool = True,
    show: bool = True,
    **kwargs: Any,
) -> Optional[BaseFigure]:
    """
    Interactive bar plot of the auto-correlation function of a time series up to a certain lag.

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param partial: Whether to use the partial auto-correlation function.
    :param n_lags: Maximum number of time steps to consider as lag.
    :param title: Plot title.
    :param show_threshold: Whether to compute the ACF significance threshold, defined as the 95% confidence upper bound
        of a time series which has no auto-correlation, and show it as a horizontal line in the plot.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    plot_df = plot_df.filter(items=[y_col])

    if n_lags is None:
        n_lags = (
            math.floor(len(plot_df) / 4)
            if partial
            else math.floor(len(plot_df) / 2)
        )

    cf = pacf if partial else acf

    if kwargs.get("alpha"):
        acf_values, conf_int = cf(df[y_col], nlags=n_lags, **kwargs)
        acf_values = acf_values[1:]

        conf_int = [np.array(value) for value in zip(*conf_int)]

        c_lower = acf_values - conf_int[0][1:]
        c_upper = conf_int[1][1:] - acf_values

    else:
        acf_values = cf(plot_df[y_col], nlags=n_lags, **kwargs)
        acf_values = acf_values[1:]

        c_lower = None
        c_upper = None

    acf_df = pd.DataFrame(
        {"lag": linspace(1, n_lags, n_lags), "acf": acf_values}
    )

    default_title = "PACF" if partial else "ACF"
    fig = _plot_plotly(
        acf_df,
        kind="bar",
        x="lag",
        y="acf",
        title=title if title else default_title,
        error_y=c_upper,
        error_y_minus=c_lower,
        x_title="Lag",
        y_title="Value",
    )

    if show_threshold:
        threshold = 2 / np.sqrt(len(acf_values))
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=acf_df["lag"],
                y=[threshold] * len(acf_values),
                line=dict(color="Red", dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                mode="lines",
                x=acf_df["lag"],
                y=[-threshold] * len(acf_values),
                line=dict(color="Red", dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if show:
        fig.show()
    else:
        return fig


def plot_psd(
    df: pd.DataFrame,
    y_col: str = None,
    time_col: str = None,
    nfft: int = None,
    fs: int = 1,
    min_period: int = 0,
    max_period: int = np.inf,
    plot_time: bool = False,
    title: str = None,
    show: bool = True,
    **kwargs: Any,
) -> Optional[BaseFigure]:
    """
    Interactive histogram of the spectral density of a time series.

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :param nfft: Length of the FFT used. If :code:`None` the length of df will be used.
    :param fs: Sampling frequency of df.
    :param min_period: Minimum period to consider.
    :param max_period: Maximum period to consider.
    :param plot_time: If :code:`True`, plot time on the x axis, else plot sampling frequency.
    :param title: Plot title.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    plot_df = plot_df.filter(items=[y_col])

    f, spectral_density = periodogram(
        plot_df[y_col], fs=fs, nfft=nfft, **kwargs
    )
    spectral_df = pd.DataFrame({"f": f, "spectral_density": spectral_density})
    spectral_df["t"] = 1 / spectral_df.f
    spectral_df = spectral_df[
        (spectral_df.t < max_period) & (spectral_df.t > min_period)
    ]
    def_title = "PSD"
    fig = _plot_plotly(
        spectral_df,
        x="t" if plot_time else "f",
        y="spectral_density",
        kind="hist",
        x_title="Period" if plot_time else "Frequency",
        y_title="Density",
        title=title if title else def_title,
    )
    if show:
        fig.show()
    else:
        return fig


def plot_ts_analysis(
    df: pd.DataFrame,
    y_col: str = None,
    time_col: str = None,
    nfft: int = 1024,
    acf_n_lags: int = None,
    pacf_n_lags: int = None,
    alpha: float = 0.1,
    show: bool = True,
    title: str = None,
) -> Optional[BaseFigure]:
    """
    Composite plot showing the time series, its spectral density, ACF and PACF.

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param nfft: Length of the FFT used. If :code:`None` the length of df will be used.
    :param acf_n_lags: Maximum number of time steps to consider as lag for ACF.
    :param pacf_n_lags: Maximum number of time steps to consider as lag for PACF.
    :param alpha: If a number is given, the confidence intervals for the given level are displayed in the plot.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :param title: Plot title.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    plot_df = plot_df.filter(items=[y_col])

    if acf_n_lags is None:
        acf_n_lags = math.floor(len(plot_df) / 2)
    if pacf_n_lags is None:
        pacf_n_lags = math.floor(len(plot_df) / 4)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Periodogram", "ACF", "PACF"],
    )
    def_title = "Time series analysis"
    fig.update_layout(
        template=template,
        showlegend=False,
        title=title if title else def_title,
    )
    fig.update_xaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)

    # --- Periodogram ---
    f, pxx = periodogram(plot_df[y_col], nfft=nfft)
    f = f[1:]
    pxx = pxx[1:]
    periodogram_df = pd.DataFrame(dict(freq=f, density=pxx))

    periodogram_trace = _plot_plotly(
        periodogram_df,
        x="freq",
        y="density",
        kind="line",
    ).data[0]

    # --- ACF ---
    acf_traces = plot_acf(
        plot_df, n_lags=acf_n_lags, alpha=alpha, show=False
    ).data

    # --- PACF ---
    pacf_traces = plot_acf(
        plot_df, n_lags=pacf_n_lags, partial=True, alpha=alpha, show=False
    ).data

    fig.add_trace(periodogram_trace, row=1, col=1)
    for trace in acf_traces:
        fig.add_trace(trace, row=2, col=1)

    for trace in pacf_traces:
        fig.add_trace(trace, row=2, col=2)

    if show:
        fig.show()
    else:
        return fig


def plot_distribution(
    df: pd.DataFrame,
    y_col: str = None,
    time_col: str = None,
    bins: int = None,
    title: str = None,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive histogram of a time series. Useful to assess marginal distribution shape.

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param bins: Number of bins in the histogram.
    :param title: Plot title.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    distr_df = plot_df.filter(items=[y_col])
    fig = _plot_plotly(
        distr_df,
        kind="histogram",
        histnorm="probability",
        nbins=bins,
        title=title,
        x_title="Value",
        y_title="Frequency",
    )
    fig.update_layout(showlegend=False)

    if show:
        fig.show()
    else:
        return fig


def plot_gof(
    df: pd.DataFrame,
    y_col: str,
    y_hat_col: str,
    time_col: str = None,
    acf_n_lags: int = None,
    pacf_n_lags: int = None,
    alpha: float = 0.1,
    title: str = "Goodness of Fit",
    subplot_titles: Tuple[str, str, str, str, str] = (
        "Actual vs Predicted Series",
        "Actual vs Predicted Scatter",
        "Residuals",
        "Residuals ACF",
        "Residuals PACF",
    ),
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive plot of goodness of fit visualizations. In order: Actual Series vs Predicted Series,
    Residuals series, and Actual vs Predicted scatter plot.

    :param acf_n_lags: Maximum number of time steps to consider as lag for ACF.
    :param pacf_n_lags: Maximum number of time steps to consider as lag for PACF.
    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param y_hat_col: Name of the column from df to be used as forecast for the time series in the plot.
    :param alpha: If a number is given, the confidence intervals for the given level are displayed in the plot.
    :param title: Plot title.
    :param subplot_titles: Tuple of titles for each of the 3 subplots.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Tuple of plotly figures if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    plot_df["Resid"] = plot_df[y_col] - plot_df[y_hat_col]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=subplot_titles,
        specs=[[{}, {"rowspan": 2}], [{}, None], [{}, {}]],
    )
    fig.update_layout(template=template, showlegend=False, title=title)
    ts_traces = plot_ts(plot_df, y_cols=[y_col, y_hat_col], show=False).data
    for trace in ts_traces:
        fig.add_trace(trace, row=1, col=1)

    res_trace = plot_ts(plot_df, y_cols=["Resid"], show=False).data[0]
    res_trace["line"]["color"] = colorway[2]
    fig.add_trace(
        res_trace,
        row=2,
        col=1,
    )

    scatter_traces = plot_scatter_fit(
        plot_df, y_col, y_hat_col, fit=True, show=False
    ).data
    for trace in scatter_traces:
        fig.add_trace(
            trace,
            row=1,
            col=2,
        )

    if acf_n_lags is None:
        acf_n_lags = math.floor(len(plot_df["Resid"].dropna()) / 2)
    if pacf_n_lags is None:
        pacf_n_lags = math.floor(len(plot_df["Resid"].dropna()) / 4)

    resid_acf_traces = plot_acf(
        plot_df.dropna(),
        y_col="Resid",
        n_lags=acf_n_lags,
        alpha=alpha,
        show=False,
    ).data
    for trace in resid_acf_traces:
        fig.add_trace(
            trace,
            row=3,
            col=1,
        )

    resid_pacf_traces = plot_acf(
        plot_df.dropna(),
        y_col="Resid",
        partial=True,
        n_lags=pacf_n_lags,
        alpha=alpha,
        show=False,
    ).data
    for trace in resid_pacf_traces:
        fig.add_trace(
            trace,
            row=3,
            col=2,
        )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text=y_col, row=1, col=2)
    fig.update_yaxes(title_text=y_hat_col, row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=2)
    fig.update_yaxes(title_text="Value", row=3, col=2)

    if show:
        fig.show()
    else:
        return fig


def plot_ts(
    df: pd.DataFrame,
    y_cols: List[str] = None,
    time_col: str = None,
    title: str = None,
    tf: Union[
        Literal["Box-Cox", "Yeo-Johnson", "log", "moving_average"],
        Callable[[Iterable], Iterable],
    ] = None,
    tf_args: Tuple = (),
    tf_kwargs: dict = None,
    keep_original: bool = True,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive plot of one or mode time series, with possibility of transforming them as well.

    :param df: Dataframe to use as the data source for the plot.
    :param y_cols: Name of the columns from df to be used as time series values in the plot. If :code:`None`, all columns are used.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param tf: If provided, a transformation to be applied to the time series. It can be one of "Box-Cox",
        "Yeo-Johnson", "log", "moving_average" or a custom callable one.
    :param tf_args: Positional arguments for the transformation. See tf for details.
    :param tf_kwargs: Keyword arguments for the transformation. See tf for details.
    :param keep_original: Whether to show the original along with the transformed one. Only active if tf is not :code:`None`.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    if tf_kwargs is None:
        tf_kwargs = {}
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    if y_cols:
        plot_df = plot_df.filter(items=y_cols)
    if tf:
        transformation = transform_dict.get(tf, tf)
        transformed_df = (
            plot_df.apply(transformation, args=tf_args, **tf_kwargs)
            .add_prefix(f"{tf}(")
            .add_suffix(")")
        )
        plot_df = (
            pd.concat([plot_df, transformed_df], axis=1)
            if keep_original
            else transformed_df
        )

    def_title = (
        "Time series (" + ", ".join(y_cols) + ")" if y_cols else "Time series"
    )
    fig = _plot_plotly(
        plot_df,
        kind="line",
        title=title if title else def_title,
        x_title="Time",
        y_title="Value",
    )
    fig.update_layout(legend_title_text="")
    if show:
        fig.show()
    else:
        return fig


def plot_seasonal_ts(
    df: pd.DataFrame,
    period: Union[
        Literal["minute", "hour", "day", "week", "month", "quarter", "year"],
        Tuple[Callable[[Iterable], Iterable]],
    ],
    y_col: str = None,
    time_col: str = None,
    title: str = None,
    subplots: bool = False,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive plot of the seasonal components of a time series.

    :param df: Dataframe to use as the data source for the plot.
    :param period: The periodicity of the seasonal component. It can be one of 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year' or a couple of callables specifying both the periodicity and the granularity of data (e.g. daily periodicity, hourly granularity).
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param subplots: Whether to split the seasonal components into subplots. Should be set to :code:`False` if the number of components is too high.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    season = period if isinstance(period, str) else "season"

    try:
        plot_df[season] = valid_seasons["grouping"].get(period, period[0])(
            plot_df.index
        )
        plot_df.index = valid_seasons["granularity"].get(period, period[1])(
            plot_df.index
        )
    except TypeError:
        print(
            "'Period' param must be either a valid string ('minute', 'hour', 'day', 'week', 'month', 'quarter', "
            "'year') or a custom function computing season from df.index."
        )
        return

    try:
        plot_df = plot_df.pivot(columns=season, values=y_col)
    except ValueError:
        print(
            "The selected seasonality is not suited for this dataframe due to its time span and/or its granularity."
            " You can try with a custom one or adjust your dataframe."
        )
        return

    def_title = f"Time series by {season}"

    try:
        fig = _plot_plotly(
            plot_df,
            kind="line",
            facet_row=season if subplots else None,
            title=title if title else def_title,
            x_title="Time",
            y_title="Value",
        )
        apply_grad_color_to_traces(
            fig, seq_colorscale_bounds[0], seq_colorscale_bounds[1]
        )

    except ValueError as e:
        print(e)
        print(
            "The selected seasonality cannot be displayed in subplots (too many traces). Try setting subplots=False."
        )
        return

    fig.update_layout(legend_title_text="")
    if show:
        fig.show()
    else:
        return fig


def plot_decomposed_ts(
    df: pd.DataFrame,
    method: Literal["STL", "seasonal_additive", "seasonal_multiplicative"],
    y_col: str = None,
    time_col: str = None,
    title: str = None,
    subplots: bool = True,
    show: bool = True,
    **decomp_kwargs: Any,
) -> Optional[BaseFigure]:
    """
    Interactive plot of a decomposition of a time series.

    :param df: Dataframe to use as the data source for the plot.
    :param method: Method of decomposition. It can be one of "STL", "seasonal_additive" or "seasonal_multiplicative".
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param subplots: Whether to split the decomposition components into subplots.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :param decomp_kwargs: Keyword arguments for the selected decomposition method. See method for details.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    y_col = y_col if y_col else plot_df.columns[0]
    plot_df = plot_df.filter(items=[y_col])

    decomp_model = decomp_methods[method]
    decomp_func = list(decomp_model.keys())[0]
    kwargs = decomp_model[decomp_func]
    kwargs.update(decomp_kwargs)
    res = decomp_func(plot_df, **kwargs)
    if decomp_func is STL:
        res = res.fit()
    components = get_components(res)
    decomposed_df = pd.DataFrame(data=components)
    if not subplots:
        for i in range(1, len(decomposed_df.columns)):
            if kwargs.get("model") == "multiplicative":
                decomposed_df.iloc[:, i] *= decomposed_df.iloc[:, i - 1]
            else:
                decomposed_df.iloc[:, i] += decomposed_df.iloc[:, i - 1]
        decomposed_df.columns = [*decomposed_df.columns[:-1], "Observed"]
    def_title = f"{method} decomposition"
    fig = _plot_plotly(
        decomposed_df,
        kind="line",
        facet_row="variable" if subplots else None,
        title=title if title else def_title,
        x_title="Time",
        y_title="Value",
    )
    fig.update_layout(legend_title_text="")
    if show:
        fig.show()
    else:
        return fig


def plot_forecast(
    df: pd.DataFrame,
    y_col: str,
    y_hat_cols: List[str],
    lower_col: str = None,
    upper_col: str = None,
    time_col: str = None,
    title: str = None,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive plot of a time series forecast (or multiple forecasts).

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param y_hat_cols: Name of the columns from df to be used as forecasts for the time series in the plot.
    :param lower_col: If given, name of the column to be used as lower confidence bound for the forecast.
    :param upper_col: If given, name of the column to be used as upper confidence bound for the forecast.
    :param title: Plot title.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    line_df = plot_df.filter(items=[y_col] + y_hat_cols)

    def_title = f"Forecast ({y_col})"
    fig = _plot_plotly(
        line_df,
        kind="line",
        title=title if title else def_title,
        x_title="Time",
        y_title="Value",
    )
    if upper_col:
        fig.add_trace(
            go.Scatter(
                name="Upper Bound",
                x=plot_df.index,
                y=plot_df[upper_col],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if lower_col:
        fig.add_trace(
            go.Scatter(
                name="Lower Bound",
                x=plot_df.index,
                y=plot_df[lower_col],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(legend_title_text="")
    if show:
        fig.show()
    else:
        return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = None,
    lags_x: List[int] = None,
    lags_y: List[int] = None,
    time_col: str = None,
    title: str = None,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Interactive scatter matrix plot of a set of variables and possibly lagged versions of them.

    :param df: Dataframe to use as the data source for the plot.
    :param x_col: Name of the column to be used as first variable in the matrix.
    :param y_col: If given, name of the column to be used as second variable in the matrix.
    :param lags_x: If given, list of lags to be applied to x_col and shown as lagged series in the matrix.
    :param lags_y: If given, list of lags to be applied to y_col and shown as lagged series in the matrix.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    if y_col is None and lags_x is None:
        raise ValueError(
            "At least one between y_col and lags_x must be provided"
        )
    else:
        sel_vars = [x_col, y_col]
        scatter_df = plot_df.filter(items=sel_vars)
        if lags_x:
            for lag in lags_x:
                scatter_df[f"{x_col} lag({lag})"] = plot_df[x_col].shift(
                    periods=lag, axis=0
                )
        if lags_y:
            for lag in lags_y:
                scatter_df[f"{y_col} lag({lag})"] = plot_df[y_col].shift(
                    periods=lag, axis=0
                )

        sel_vars_string = ", ".join(sel_vars) if y_col else x_col

        feats = list(scatter_df.columns)
        n = len(feats)
        feats_cols = list(feats)[:-1]
        feats_rows = list(feats)[1:]

        fig = make_subplots(
            rows=n - 1,
            cols=n - 1,
            column_titles=feats_cols,
            row_titles=feats_rows,
        )
        def_title = f"Scatter matrix ({sel_vars_string}) with lags"
        fig.update_layout(
            template=template,
            showlegend=False,
            title=title if title else def_title,
        )

        for x_var, y_var in product(feats_cols, feats_rows):
            x_col = scatter_df[x_var]
            y_col = scatter_df[y_var]
            if x_var != y_var:
                i = feats_rows.index(y_var)
                j = feats_cols.index(x_var)
                # --- Scatterplot ---
                scatter_trace = plot_scatter_fit(
                    df=scatter_df, x_col=x_col, y_col=y_col, show=False
                ).data[0]
                fig.add_trace(scatter_trace, row=i + 1, col=j + 1)

        if show:
            fig.show()
        else:
            return fig


def plot_scatter_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str = None,
    title: str = None,
    fit: Union[bool, Literal["summary"]] = False,
    show: bool = True,
    **kwargs,
) -> Optional[BaseFigure]:
    """
    Interactive scatter plot between two time series with possibility of showing a fit line between them.

    :param df: Dataframe to use as the data source for the plot.
    :param x_col: Name of the column from df to be used as x variable in the scatter plot.
    :param y_col: Name of the column from df to be used as y variable in the scatter plot.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param fit: Whether to show the fit line in the plot. If set to "summary", displays fit statistics at the bottom.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :param kwargs: Keyword arguments for fit, if not set to :code:`False`. They are passed to :code:`pandas.plot`.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    if fit is not False:
        fit_dict = {"trendline": "ols"}
        fit_dict.update(**kwargs)
        kwargs = fit_dict

    def_title = f"Scatter plot ({x_col} vs {y_col})"
    fig = _plot_plotly(
        plot_df,
        kind="scatter",
        x=x_col,
        y=y_col,
        title=title if title else def_title,
        x_title=str(x_col),
        y_title=str(y_col),
        **kwargs,
    )
    if show:
        fig.show()
        if kwargs.get("trendline") == "ols" and fit == "summary":
            results = px.get_trendline_results(fig)
            results = results.iloc[0]["px_fit_results"].summary()
            print(results)
    else:
        if kwargs.get("trendline") == "ols" and fit == "summary":
            results = px.get_trendline_results(fig)
            results = results.iloc[0]["px_fit_results"].summary()
            return fig, results
        else:
            return fig


def plot_inverse_arma_roots(
    process: ArmaProcess, show: bool = True
) -> Optional[BaseFigure]:
    """
    Plot of the inverse roots of a given ARMA process.

    :param process: The ARMA process (statsmodels.tsa.ArmaProcess) for which the roots are computed.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    copied_process = deepcopy(process)
    roots = copied_process.arroots
    inv_roots = 1 / roots
    re = [root.real for root in inv_roots]
    im = [root.imag for root in inv_roots]
    inv_roots_df = pd.DataFrame(
        {
            "Root": [str(np.round(r, 5))[1:-1] for r in inv_roots],
            "Re": re,
            "Im": im,
        }
    )

    layout = dict(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
            dtick=0.25,
            gridwidth=0.1,
            gridcolor="grey",
            zerolinecolor="grey",
            zerolinewidth=2,
            title="Im(1/root)",
        ),
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            dtick=0.25,
            gridwidth=0.1,
            gridcolor="grey",
            zerolinecolor="grey",
            zerolinewidth=2,
            title="Re(1/root)",
        ),
    )

    fig = _plot_plotly(
        inv_roots_df, kind="scatter", x=re, y=im, title=f"Inverse ARMA roots"
    )
    fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1)
    fig.update_layout(layout)

    if show:
        fig.show()
    else:
        return fig


def plot_extended_scatter_matrix(
    df: pd.DataFrame,
    time_col: str = None,
    y_cols: List[str] = None,
    title: str = None,
    show: bool = True,
) -> Optional[BaseFigure]:
    """
    Composite plot containing scatter matrix, distributions and couple-correlations of a set of time series.

    :param df: Dataframe to use as the data source for the plot.
    :param y_cols: Name of the columns from df to be used as variables in the plot. If :code:`None`, all columns are used.
    :param time_col: Name of the column from df to be used as time variable in the plot.
    :param title: Plot title.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    set_time_index(plot_df, time_col)
    if y_cols:
        plot_df = plot_df.filter(items=y_cols)
    feats = plot_df.columns
    indices = list(range(len(feats)))
    n = len(feats)
    fig = make_subplots(
        rows=n, cols=n, row_titles=list(feats), column_titles=list(feats)
    )
    def_title = "Scatter Matrix, extended"
    fig.update_layout(
        template=template,
        showlegend=False,
        title=title if title else def_title,
    )

    for i, j in product(indices, indices):
        x = plot_df[feats[i]]
        y = plot_df[feats[j]]
        if i > j:
            # --- Scatterplot ---
            scatter_trace = plot_scatter_fit(
                plot_df, feats[i], feats[j], show=False
            ).data[0]
            fig.add_trace(scatter_trace, row=i + 1, col=j + 1)
        elif i < j:
            # --- Correlation coefficient ---
            fig.add_trace(
                go.Scatter(
                    x=[1],
                    y=[1],
                    mode="text",
                    text=f"Corr: {round(pearsonr(x, y)[0], 2)}",
                    textposition="middle center",
                    hoverinfo="skip",
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.update_xaxes(
                showgrid=False,
                visible=False,
                zeroline=False,
                row=i + 1,
                col=j + 1,
            )
            fig.update_yaxes(
                showgrid=False,
                visible=False,
                zeroline=False,
                row=i + 1,
                col=j + 1,
            )
        else:
            # --- Distribution ---
            trace = plot_distribution(
                plot_df, y_col=feats[i], show=False
            ).data[0]
            fig.add_trace(trace, row=i + 1, col=j + 1)

    if show:
        fig.show()
    else:
        return fig


def plot_ts_overview(
    df: pd.DataFrame,
    y_col: str = None,
    n_lags: int = None,
    alpha: float = 0.1,
    show: bool = True,
    title: str = None,
) -> Optional[BaseFigure]:
    """
    Composite plot showing a time series, its auto-correlation function and its distribution.

    :param df: Dataframe to use as the data source for the plot.
    :param y_col: Name of the column from df to be used as time series values in the plot.
    :param n_lags: Maximum number of time steps to consider as lag.
    :param alpha: If a number is given, the confidence intervals for the given level are displayed in the plot.
    :param show: Whether to call :code:`Figure.show` at the end of the function. If :code:`False`, returns the figure.
    :param title: Plot title.
    :return: Plotly figure if show is :code:`True`, else nothing.
    """
    plot_df = df.copy()
    y_col = y_col if y_col else plot_df.columns[0]
    plot_df = plot_df.filter(items=[y_col])

    if n_lags is None:
        n_lags = math.floor(len(plot_df) / 2)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Time series", "ACF", "Distribution"],
    )
    def_title = "Time series overview"
    fig.update_layout(
        template=template,
        showlegend=False,
        title=title if title else def_title,
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)

    # --- Time series ---
    ts_trace = plot_ts(plot_df, y_cols=[y_col], show=False).data[0]

    # --- ACF ---
    acf_traces = plot_acf(
        plot_df, y_col=y_col, n_lags=n_lags, alpha=alpha, show=False
    ).data

    # --- Distribution ---
    dist_trace = plot_distribution(plot_df, y_col=y_col, show=False).data[0]

    fig.add_trace(ts_trace, row=1, col=1)
    for trace in acf_traces:
        fig.add_trace(trace, row=2, col=1)

    fig.add_trace(dist_trace, row=2, col=2)

    if show:
        fig.show()
    else:
        return fig
