# pytsviz

![GitHub](https://img.shields.io/github/license/xtreamsrl/pytsviz)
![GitHub issues](https://img.shields.io/github/issues/xtreamsrl/pytsviz)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xtreamsrl/pytsviz/blob/master/docs/source/notebooks/data_visualization_examples.ipynb)
[![Downloads](https://pepy.tech/badge/pytsviz)](https://pepy.tech/project/pytsviz)
[![Documentation Status](https://readthedocs.org/projects/pytsviz/badge/?version=latest)](https://pytsviz.readthedocs.io/en/latest/?badge=latest)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xtreamsrl/pytsviz/CI?label=tests)

*pytsviz* is a suite of tools to quickly analyze and visualize time series data. It is partially based on the [*tsviz*](https://github.com/xtreamsrl/tsviz) R package.

The *utils* module contains a set of useful utilities, not strictly related to visualization, we often use (e.g. harmonics computation).

The *viz* module contains functions for plotting univariate time series, as well as performing quick qualitative analyses such as decompositions, correlations and so on.

Some visualizations mimic the R packages *ggplot2* and *forecast*, as presented in the textbook *Forecasting: principles and practice* by Rob J. Hyndman and George Athanasopoulos.
The online version of the text can be found [here](https://otexts.com/fpp3/).

## Install

The preferred way to install the package is using pip, but you can also download the code and install from source

To install the package using pip:

```shell
pip install pytsviz
```

## Develop

After cloning, you need to install and setup Poetry. See [instructions](https://github.com/python-poetry/poetry#installation).

Then, inside the project directory, run:

```shell
poetry install
pre-commit install
```

Then, you're good to go.

You're free to submit your pull requests. Just make sure they follow [conventional commit rules](https://www.conventionalcommits.org/en/v1.0.0/#specification). This can be enforced by the [*commitizen*](https://commitizen-tools.github.io/commitizen/) tool, which is also included among the package dependencies.

Please also make sure that function documentation is consistent. We are currently using [Sphinx docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).


## Who we are
<img align="left" width="80" height="80" src="https://avatars2.githubusercontent.com/u/38501645?s=450&u=1eb7348ca81f5cd27ce9c02e689f518d903852b1&v=4">
A proudly 🇮🇹 software development and data science startup.<br>We consider ourselves a family of talented and passionate people building their own products and powerful solutions for our clients. Get to know us more on <a target="_blank" href="https://xtreamers.io">xtreamers.io</a> or follow us on <a target="_blank" href="https://it.linkedin.com/company/xtream-srl">LinkedIn</a>.