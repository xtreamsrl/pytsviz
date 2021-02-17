"""
The *viz* module contains functions to visualize most of the key aspects of a univariate time series such as (partial) correlograms, periodograms, line plots, ...
"""
from itertools import product
from typing import List, Callable, Iterable, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import gridspec
from matplotlib import pyplot as plt
from numpy.core import linspace
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import periodogram
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa._stl import STL
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.stattools import acf, pacf

from pytsviz.utils import transform_dict, set_time_index, decompose_dict, get_components

plt.rcParams["figure.figsize"] = (10, 6)
DEFAULT_LAYOUT = {
    "xaxis": {"showgrid": False, "zeroline": False},
    "yaxis": {"zeroline": False},
}

template = dict(
    layout=go.Layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            zeroline=False
        ),
        title_font=dict(
            family="Rockwell",
            size=24)
    )
)


def _plot_plotly(
    df,
    kind,
    x_title="",
    y_title="",
    x_type="-",
    y_type="-",
    layout=None,
    **kwargs
):
    with pd.option_context("plotting.backend", "plotly"):
        fig = df.plot(kind=kind, **kwargs)  # render_mode="SVG"
    template["layout"]["xaxis"]["title_text"] = x_title
    template["layout"]["yaxis"]["title_text"] = y_title
    template["layout"]["xaxis"]["type"] = x_type
    template["layout"]["yaxis"]["type"] = y_type
    fig.update_layout(template=template)
    fig.update_layout(layout)
    return fig


def plotly_acf(series, nlags, title="ACF", show_threshold=True, show=True, **kwargs):
    """
    Interactive barplot of the autocorrelation function of a time series up to a certain lag

    :param series: Time series
    :type series: `array-like`
    :param nlags: Maximum lag to consider
    :type nlags: `int`
    :param title: Plot Title
    :type title: `str`, default *"ACF"*
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """
    if kwargs.get("alpha") is not None:
        acf_values, conf_int = acf(series, nlags=nlags, **kwargs)
        acf_values = acf_values[1:]

        conf_int = [np.array(x) for x in zip(*conf_int)]

        c_lower = acf_values - conf_int[0][1:]
        c_upper = conf_int[1][1:] - acf_values

    else:
        acf_values = acf(series, nlags=nlags, **kwargs)
        acf_values = acf_values[1:]

        c_lower = None
        c_upper = None

    plot_df = pd.DataFrame(
        {
            "lag": linspace(1, nlags, nlags),
            "acf": acf_values
        }
    )

    fig = _plot_plotly(
        plot_df,
        kind="bar",
        x="lag",
        y="acf",
        title=title,
        error_y=c_upper,
        error_y_minus=c_lower
    )

    if show_threshold:
        threshold = 2 / np.sqrt(len(acf_values))
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=plot_df["lag"],
                y=[threshold] * len(acf_values),
                line=dict(color='Red', dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=plot_df["lag"],
                y=[- threshold] * len(acf_values),
                line=dict(color='Red', dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    if show:
        fig.show()
    else:
        return fig


def plotly_pacf(series, nlags, title="PACF", show_threshold=True, show=True, **kwargs):
    """
    Interactive barplot of the partial autocorrelation function of a time series up to a certain lag

    :param series: Time series
    :type series: `array-like`
    :param nlags: Maximum lag to consider
    :type nlags: `int`
    :param title: Plot Title
    :type title: `str`, default *"ACF"*
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """
    if kwargs.get("alpha") is not None:
        pacf_values, conf_int = pacf(series, nlags=nlags, **kwargs)
        pacf_values = pacf_values[1:]

        conf_int = [np.array(x) for x in zip(*conf_int)]

        c_lower = pacf_values - conf_int[0][1:]
        c_upper = conf_int[1][1:] - pacf_values

    else:
        pacf_values = pacf(series, nlags=nlags, **kwargs)
        pacf_values = pacf_values[1:]

        c_lower = None
        c_upper = None

    plot_df = pd.DataFrame(
        {
            "lag": linspace(1, nlags, nlags),
            "pacf": pacf_values
        }
    )

    fig = _plot_plotly(
        plot_df,
        kind="bar",
        x="lag",
        y="pacf",
        title=title,
        error_y=c_upper,
        error_y_minus=c_lower
    )

    if show_threshold:
        threshold = 2 / np.sqrt(len(pacf_values))
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=plot_df["lag"],
                y=[threshold] * len(pacf_values),
                line=dict(color='Red', dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                mode='lines',
                x=plot_df["lag"],
                y=[- threshold] * len(pacf_values),
                line=dict(color='Red', dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    if show:
        fig.show()
    else:
        return fig


def plotly_psd(
    series,
    nfft=None,
    fs=1,
    min_period=0,
    max_period=np.inf,
    plot_time=False,
    title="PSD",
    **kwargs
):
    """
    Interactive histogram of the spectral density of a time series

    :param series: Time series
    :type series: `array-like`
    :param nfft: Length of the FFT used. If *None* the length of `series` will be used.
    :type nfft: `int`, optional, default *None*
    :param fs: Sampling frequency of `series`.
    :type fs: `float`, default 1
    :param min_period: Minimum period to consider
    :type min_period: `float`, default 0
    :param max_period: Maximum period to consider
    :type max_period: `float`, default `np.inf`
    :param plot_time: If *True*, plot time on the *x* axis, else plot sampling frequency.
    :type plot_time: `bool`, default *True*
    :param title: Plot title
    :type title: `str`, default *"PSD"*.
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """

    f, spectral_density = periodogram(series, fs=fs, nfft=nfft, **kwargs)
    plt_df = pd.DataFrame({"f": f, "spectral_density": spectral_density})
    plt_df["t"] = 1 / plt_df.f
    plt_df = plt_df[(plt_df.t < max_period) & (plt_df.t > min_period)]
    return _plot_plotly(
        plt_df,
        x="t" if plot_time else "f",
        y="spectral_density",
        kind="hist",
        x_title="Period" if plot_time else "Frequency",
        title=title,
    )


def tsdisplay(
    series,
    nfft=None,
    lags=None,
    plot_time=True,
    title="Time series analysis",
):
    """
    Comprehensive matplotlib plot showing: line plot of time series, spectral density, ACF and PACF.

    :param series: Time series
    :type series: `array-like`
    :param nfft: Length of the FFT used. If *None* the length of `series` will be used.
    :type nfft: `int`, optional, default *None*
    :param lags: Number of lags to compute ACF and PACF
    :type lags: `int`, default 192
    :param plot_time: If *True*, plot time on the *x* axis for spectral density, else plot sampling frequency.
    :type plot_time: `bool`, default *True*
    :param title: Plot title
    :type title: `str`, default *"Time series analysis"*
    :return: Matplotlib figure
    :rtype: :py:class:`matplotlib.figure.Figure`
    """
    if lags is None:
        lags = int(len(series) / 2 - 1)
    series = pd.Series(series)
    gs = gridspec.GridSpec(3, 2)
    fig = plt.figure()
    plt.subplot(gs[0, :])
    series.plot(title=title)
    plt.subplot(gs[1, :])
    f, pxx = periodogram(series, nfft=nfft)
    f = f[1:]
    t = 1 / f
    pxx = pxx[1:]
    if plot_time:
        plt.plot(t, pxx)
    else:
        plt.plot(f, pxx)
    plt.title("Periodogram")
    plt.subplot(gs[2, 0])
    plot_acf(series, ax=plt.gca(), lags=lags, marker=None)
    plt.subplot(gs[2, 1])
    plot_pacf(series, ax=plt.gca(), lags=lags, marker=None)
    plt.tight_layout()
    plt.close()
    return fig


def plotly_tsdisplay(
        series,
        nfft=1024,
        lags=None,
        alpha=0.1,
        show=True
):

    """
    Comprehensive plotly plot showing: line plot of time series, spectral density, ACF and PACF.

    :param series: Time series
    :type series: `array-like`
    :param nfft: Length of the FFT used. If *None* the length of `series` will be used.
    :type nfft: `int`, optional, default *None*
    :param lags: Number of lags to compute ACF and PACF
    :type lags: `int`, default 192
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """
    if lags is None:
        lags = int(len(series) / 2 - 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Periodogram", "ACF", "PACF"],
    )
    fig.update_layout(
        template=template,
        showlegend=False
    )

    # --- Periodogram ---
    f, pxx = periodogram(series, nfft=nfft)
    f = f[1:]
    t = 1 / f
    pxx = pxx[1:]

    periodogram_trace = go.Scatter(x=f, y=pxx, marker=dict(color="royalblue"))

    # --- ACF ---
    acf_traces = plotly_acf(series, nlags=lags, alpha=alpha, show=False).data

    # --- PACF ---
    pacf_traces = plotly_pacf(series, nlags=lags, alpha=alpha, show=False).data

    fig.append_trace(
        periodogram_trace,
        row=1,
        col=1
    )
    for trace in acf_traces:
        fig.append_trace(
            trace,
            row=2,
            col=1
        )

    for trace in pacf_traces:
        fig.append_trace(
            trace,
            row=2,
            col=2
        )

    if show:
        fig.show()
    else:
        return fig


def plot_distribution_histogram(series, bins=None, title="", show=True):
    """
    Plotly histogram of a time series. Useful to assess marginal distribution shape.

    :param series: Time series
    :type series: Array-like
    :param bins: Number of bins in the histogram
    :type bins: `int`, default 10
    :param title: Plot Title
    :type title: `str`, default ""
    :param color: Histogram color, check Plotly docs for accepted values
    :type color: `str`, default "royalblue"
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """

    fig = _plot_plotly(
        series,
        kind="histogram",
        histnorm="probability",
        nbins=bins,
        title=title
    )

    if show:
        fig.show()
    else:
        return fig


def plot_gof(
        fc_df,
        y_col: str,
        y_hat_col: str,
        time_col: str = None,
        title="Goodness of Fit",
        subplot_titles=(
            "Actual vs Predicted Series",
            "Residuals",
            "Actual vs Predicted Scatter",
        ),
        show=True
):
    """
    Shows an interactive plot of goodness of fit visualizations. In order: Actual Series vs Predicted Series,
    Residuals series, and Actual vs Predicted scatter plot.

    :param y: Actual series
    :type y: `array-like`
    :param y_hat: Predicted series
    :type y_hat: Array-like
    :param title: Plot Title
    :type title: `str`, default *"Goodness of Fit"*
    :param actual_name: String to name *actual* series. Shown in axis labels and legend.
    :type actual_name: `str`, default *"Actual"*
    :param predicted_name: String to name *predicted* series. Shown in axis labels and legend.
    :type predicted_name: `str`, default *"Actual"*
    :param subplot_titles: Tuple of titles for each of the 3 subplots.
    :type subplot_titles: `tuple(str, str, str)`, default (*'Actual vs Predicted Series'*, *'Residuals'*,
     *Actual vs Predicted Scatter'*)
    :return: Tuple of plotly figures
    :rtype: `tuple` of :py:class:`plotly.graph_objects.Figure`. Length 2.
    """
    df = fc_df.copy()
    set_time_index(df, time_col)
    df["Resid"] = df[y_col] - df[y_hat_col]

    fig = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles)
    fig.update_layout(
        template=template,
        showlegend=False,
        title=title
    )
    ts_traces = time_series_plot(df, y_cols=[y_col, y_hat_col], show=False).data
    for trace in ts_traces:
        fig.append_trace(
            trace,
            row=1,
            col=1
        )

    res_trace = time_series_plot(df, y_cols=["Resid"], show=False).data[0]
    fig.append_trace(
        res_trace,
        row=2,
        col=1
    )

    scatter_traces = scatterplot(df, y_col, y_hat_col, fit=True, show=False).data
    for trace in scatter_traces:
        fig.append_trace(
            trace,
            row=3,
            col=1
        )
    fig.update_xaxes(title_text=y_col, row=3, col=1)
    fig.update_yaxes(title_text=y_hat_col, row=3, col=1)

    if show:
        fig.show()
    else:
        return fig


def time_series_plot(
        ts_df: pd.DataFrame,
        y_cols: List[str] = None,
        time_col: str = None,
        title: str = None,
        tf: str = None,
        custom_tf: Callable[[Iterable], Iterable] = None,
        tf_args: Tuple = (),
        tf_kwargs: dict = None,
        keep_original: bool = True,
        show=True
):
    if tf_kwargs is None:
        tf_kwargs = {}
    df = ts_df.copy()
    set_time_index(df, time_col)
    if y_cols is not None:
        df = df.filter(items=y_cols)
    if tf is not None:
        transformation = transform_dict[tf]
        tf_func = list(transformation.keys())[0]
        kwargs = transformation[tf_func]
        kwargs.update(tf_kwargs)
        transformed_df = df.apply(tf_func, args=tf_args, **kwargs).add_prefix(f"{tf}(").add_suffix(")")
        df = pd.concat([df, transformed_df], axis=1) if keep_original else transformed_df
    if custom_tf is not None:
        transformed_df = df.apply(custom_tf, args=tf_args, **kwargs).add_prefix('tf(').add_suffix(")")
        df = pd.concat([df, transformed_df], axis=1) if keep_original else transformed_df
    def_title = "Time series (" + ", ".join(y_cols) + ")"
    fig = _plot_plotly(
        df,
        kind="line",
        title=title if title is not None else def_title
    )
    if show:
        fig.show()
    else:
        return fig


def seasonal_time_series_plot(
        ts_df: pd.DataFrame,
        time_col: str = None,
        period=1,
        show=True
):
    df = ts_df.copy()
    set_time_index(df, time_col)
    beginning, duration, wrap = period
    diff_index = [t.to_pydatetime() - beginning for t in df.index]
    zipped_division = [divmod(diff, duration) for diff in diff_index]
    q_list, r_list = list(zip(*zipped_division))
    df["period"] = q_list
    df["new_index"] = [r.days for r in r_list]
    df = df.pivot(index="new_index", columns="period", values="y")

    fig = _plot_plotly(
        df,
        kind="line"
    )
    if show:
        fig.show()
    else:
        return fig


def decomposed_time_series_plot(
        ts_df: pd.DataFrame,
        time_col: str = None,
        title: str = None,
        method: str = "STL",
        subplots: bool = True,
        cumulative: bool = True,
        show=True,
        **decomp_kwargs
):
    df = ts_df.copy()
    set_time_index(df, time_col)
    decomp_model = decompose_dict[method]
    decomp_func = list(decomp_model.keys())[0]
    kwargs = decomp_model[decomp_func]
    kwargs.update(decomp_kwargs)
    res = decomp_func(df, **kwargs)
    if decomp_func is STL:
        res = res.fit()
    if decomp_func is UnobservedComponents:
        res = res.fit()
    components = get_components(res)
    df = pd.DataFrame(data=components)
    if not subplots and cumulative:
        for i in range(1, len(df.columns)):
            if kwargs.get("model") == "multiplicative":
                df.iloc[:, i] *= df.iloc[:, i - 1]
            else:
                df.iloc[:, i] += df.iloc[:, i - 1]
        df.columns = [*df.columns[:-1], 'Observed']
    def_title = f"{method} decomposition"
    fig = _plot_plotly(
        df,
        kind="line",
        facet_row='variable' if subplots else None,
        title=title if title is not None else def_title
    )
    if show:
        fig.show()
    else:
        return fig


def forecast_plot(
        fc_df: pd.DataFrame,
        ts_col: str,
        fc_cols: List[str],
        lower_col: str = None,
        upper_col: str = None,
        time_col: str = None,
        title: str = None,
        show=True
):
    df = fc_df.copy()
    set_time_index(df, time_col)
    line_df = df.filter(items=[ts_col] + fc_cols)

    def_title = f"Forecast ({ts_col})"
    fig = _plot_plotly(
        line_df,
        kind="line",
        title=title if title is not None else def_title
    )
    if upper_col is not None:
        fig.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=fc_df.index,
                y=fc_df[upper_col],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            row=1,
            col=1
        )
    if lower_col is not None:
        fig.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=fc_df.index,
                y=fc_df[lower_col],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ),
            row=1,
            col=1
        )

    if show:
        fig.show()
    else:
        return fig


def vars_scatterplot(
        feat_df,
        var1: str,
        var2: str = None,
        lags1: List[int] = None,
        lags2: List[int] = None,
        time_col: str = None,
        title: str = None,
        show=True
):
    df = feat_df.copy()
    set_time_index(df, time_col)
    if var2 is None and lags1 is None:
        print("At least one between var2 and lags1 must be provided")
        return
    else:
        sel_vars = [var1, var2]
        scatter_df = df.filter(items=sel_vars)  # if var2 is None else df.filter(items=[var1, var2])
        if lags1 is not None:
            for lag in lags1:
                scatter_df[f"{var1} lag({lag})"] = df[var1].shift(periods=lag, axis=0)
        if lags2 is not None:
            for lag in lags2:
                scatter_df[f"{var2} lag({lag})"] = df[var2].shift(periods=lag, axis=0)

        sel_vars_string = ", ".join(sel_vars) if var2 is not None else var1

        feats = list(scatter_df.columns)
        n = len(feats)
        feats_cols = list(feats)[:-1]
        feats_rows = list(feats)[1:]

        fig = make_subplots(
            rows=n - 1,
            cols=n - 1,
            column_titles=feats_cols,
            row_titles=feats_rows
        )
        def_title = f"Scatter matrix ({sel_vars_string}) with lags"
        fig.update_layout(
            template=template,
            showlegend=False,
            title=title if title is not None else def_title
        )

        for x_var, y_var in product(feats_cols, feats_rows):
            x = scatter_df[x_var]
            y = scatter_df[y_var]
            if x_var != y_var:
                i = feats_rows.index(y_var)
                j = feats_cols.index(x_var)
                # --- Scatterplot ---
                scatter_trace = scatterplot(
                    scatter_df,
                    x,
                    y,
                    show=False
                ).data[0]
                fig.add_trace(
                    scatter_trace,
                    row=i + 1,
                    col=j + 1
                )

        if show:
            fig.show()
        else:
            return fig


def scatterplot(
        feat_df,
        var1: str,
        var2: str,
        time_col: str = None,
        title: str = None,
        fit=False,
        show_stats_summary: bool = False,
        show=True,
        **kwargs
):
    df = feat_df.copy()
    set_time_index(df, time_col)
    if fit:
        fit_dict = {"trendline": "ols"}
        fit_dict.update(**kwargs)
        kwargs = fit_dict

    def_title = f"Scatter plot ({var1} vs {var2})"
    fig = _plot_plotly(
        df,
        kind="scatter",
        x=var1,
        y=var2,
        title=title if title is not None else def_title,
        **kwargs
    )
    if show:
        fig.show()
        if kwargs.get("trendline") == "ols" and show_stats_summary:
            results = px.get_trendline_results(fig)
            results = results.iloc[0]["px_fit_results"].summary()
            print(results)
    else:
        if kwargs.get("trendline") == "ols" and show_stats_summary:
            results = px.get_trendline_results(fig)
            results = results.iloc[0]["px_fit_results"].summary()
            return fig, results
        else:
            return fig


def inverse_arma_roots_plot(
    process: ArmaProcess,
    show=True
):
    roots = process.arroots
    re = [x.real for x in roots]
    im = [x.imag for x in roots]
    roots_df = pd.DataFrame({"Root": [str(round(r, 5))[1:-1] for r in roots], "Re": re, "Im": im})

    layout = dict(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
            dtick=0.5,
            gridwidth=0.1,
            gridcolor='grey',
            zerolinecolor='grey',
            zerolinewidth=2,
            title="Im"
        ),
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            dtick=0.5,
            gridwidth=0.1,
            gridcolor='grey',
            zerolinecolor='grey',
            zerolinewidth=2,
            title="Re"
        )
    )

    fig = _plot_plotly(
        roots_df,
        kind="scatter",
        x=re,
        y=im,
        title=f"Inverse arma roots",
        hover_name="Root",
        layout=layout
    )
    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-1, y0=-1, x1=1, y1=1
                  )

    if show:
        fig.show()
    else:
        return fig


def composite_matrix_scatterplot(
        feat_df,
        time_col: str = None,
        y_cols: List[str] = None
):
    df = feat_df.copy()
    set_time_index(df, time_col)
    if y_cols is not None:
        df = df.filter(items=y_cols)
    feats = df.columns
    indices = list(range(len(feats)))
    n = len(feats)
    fig = make_subplots(
        rows=n,
        cols=n,
        row_titles=list(feats),
        column_titles=list(feats)
    )
    fig.update_layout(
        template=template,
        showlegend=False
    )

    for i, j in product(indices, indices):
        x = df[feats[i]]
        y = df[feats[j]]
        if i > j:
            # --- Scatterplot ---
            scatter_trace = scatterplot(feat_df, x, y, show=False).data[0]
            fig.add_trace(
                scatter_trace,
                row=i+1,
                col=j+1
            )
        elif i < j:
            # --- Correlation coefficient ---
            fig.add_trace(
                go.Scatter(
                    x=[1],
                    y=[1],
                    mode="text",
                    text=f"Corr: {round(pearsonr(x, y)[0], 2)}",
                    textposition="middle center",
                    hoverinfo="skip"
                ),
                row=i + 1,
                col=j + 1
            )
            fig.update_xaxes(
                showgrid=False,
                visible=False,
                zeroline=False,
                row=i + 1,
                col=j + 1
            )
            fig.update_yaxes(
                showgrid=False,
                visible=False,
                zeroline=False,
                row=i + 1,
                col=j + 1
            )
        else:
            # --- Distribution ---
            trace = plot_distribution_histogram(x, show=False).data[0]
            fig.add_trace(trace, row=i + 1, col=j + 1)
    fig.show()


def composite_summary_plot(
        series,
        lags=None,
        alpha=0.1,
        show=True
):

    if lags is None:
        lags = int(len(series) / 2 - 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=["Time series", "ACF", "Distribution"],
    )
    fig.update_layout(
        template=template,
        showlegend=False
    )

    # --- Time series ---
    ts_trace = time_series_plot(series, show=False).data[0]

    # --- ACF ---
    acf_traces = plotly_acf(series, nlags=lags, alpha=alpha, show=False).data

    # --- Distribution ---
    dist_trace = plot_distribution_histogram(series, show=False).data[0]

    fig.append_trace(
        ts_trace,
        row=1,
        col=1
    )
    for trace in acf_traces:
        fig.append_trace(
            trace,
            row=2,
            col=1
        )

    fig.append_trace(
        dist_trace,
        row=2,
        col=2
    )

    if show:
        fig.show()
    else:
        return fig
