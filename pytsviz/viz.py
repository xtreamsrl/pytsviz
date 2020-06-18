import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

plt.rcParams["figure.figsize"] = (10, 6)
DEFAULT_LAYOUT = {
    "xaxis": {"showgrid": False, "zeroline": False},
    "yaxis": {"zeroline": False},
}


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
        fig = df.plot(kind=kind, render_mode="SVG", **kwargs)
    if layout is None:
        layout = DEFAULT_LAYOUT.copy()
    layout["xaxis"]["title_text"] = x_title
    layout["yaxis"]["title_text"] = y_title
    layout["xaxis"]["type"] = x_type
    layout["yaxis"]["type"] = y_type
    fig.update_layout(layout)
    return fig


def plotly_acf(series, nlags, title="ACF"):
    """
    Interactive barplot of the autocorrelation function of a time series up to a certain lag

    :param series: Time series
    :type series: `~numpy.array-like`
    :param nlags: Maximum lag to consider
    :type nlags: `int`
    :param title: Plot Title
    :type title: `str`, default *"ACF"*
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """
    acf_values = acf(series, nlags=nlags)
    return _plot_plotly(
        pd.DataFrame({"acf": acf_values}),
        kind="bar",
        colors=["blue"],
        title=title,
    )


def plotly_pacf(series, nlags, title="PACF"):
    """
    Interactive barplot of the partial autocorrelation function of a time series up to a certain lag

    :param series: Time series
    :type series: `~numpy.array-like`
    :param nlags: Maximum lag to consider
    :type nlags: `int`
    :param title: Plot Title
    :type title: `str`, default *"PACF"*
    :return: Plotly figure
    :rtype: :py:class:`plotly.basedatatypes.BaseFigure`
    """
    acf_values = pacf(series, nlags=nlags)
    return _plot_plotly(
        pd.DataFrame({"pacf": acf_values}),
        kind="bar",
        colors=["blue"],
        title=title,
    )


def plotly_psd(
    series,
    nfft=None,
    fs=1,
    min_period=0,
    max_period=np.inf,
    plot_time=True,
    title="PSD",
):
    """
    Interactive histogram of the spectral density of a time series

    :param series: Time series
    :type series: `~numpy.array-like`
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

    f, spectral_density = periodogram(series, fs=fs, nfft=nfft)
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
    lags=8 * 24,
    plot_time=True,
    title="Time series analysis",
):
    """
    Comprehensive matplotlib plot showing: line plot of time series, spectral density, ACF and PACF.

    :param series: Time series
    :type series: `~numpy.array-like`
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


def plot_gof(
    y,
    y_hat,
    title="Goodness of Fit",
    actual_name="Actual",
    predicted_name="Predicted",
    subplot_titles=(
        "Actual vs Predicted Series",
        "Residuals",
        "Actual vs Predicted Scatter",
    ),
):
    """
    Shows an interactive plot of goodness of fit visualizations. In order: Actual Series vs Predicted Series,
    Residuals series, and Actual vs Predicted scatterplot.

    :param y: Actual series
    :type y: Array-like
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
    :rtype: :py:class:`plotly.graph_objects.Figure`
    """

    fig = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles)
    res = y - y_hat
    df = (
        pd.DataFrame({actual_name: y, predicted_name: y_hat, "res": res})
        .dropna()
        .sort_index()
    )
    series_fig = _plot_plotly(
        df,
        y=[actual_name, predicted_name],
        kind="line",
        title=subplot_titles[0],
    )
    fig.append_trace(series_fig["data"][0], 1, 1)
    fig.append_trace(series_fig["data"][1], 1, 1)
    res_fig = _plot_plotly(df, y="res", kind="line", title=subplot_titles[1])
    fig.append_trace(res_fig["data"][0], 2, 1)
    actual_vs_predicted_fig = _plot_plotly(
        df,
        x=actual_name,
        y=predicted_name,
        kind="scatter",
        x_title=actual_name,
        y_title=predicted_name,
        opacity=0.5,
        title=subplot_titles[2],
    )
    fig.append_trace(actual_vs_predicted_fig["data"][0], 3, 1)
    fig.update_xaxes(title_text=actual_name, row=3, col=1)
    fig.update_yaxes(title_text=predicted_name, row=3, col=1)
    fig.update_layout(title_text=title)
    fig.show()
    return actual_vs_predicted_fig, res_fig, series_fig
