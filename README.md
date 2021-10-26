# pytsviz

![GitHub](https://img.shields.io/github/license/xtreamsrl/pytsviz)
![GitHub issues](https://img.shields.io/github/issues/xtreamsrl/pytsviz)

*pytsviz* is a suite of tools to quickly analyze and visualize time series data. It is partially based on the [*tsviz*](https://github.com/xtreamsrl/tsviz) R package.

The *utils* module contains a set of useful utilities, not strictly related to visualization, we often use (e.g. harmonics computation).

The *viz* module contains functions for plotting univariate time series, as well as performing quick qualitative analyses such as decompositions, correlations and so on.

Some visualizations mimic the R packages *ggplot2* and *forecast*, as presented in the textbook *Forecasting: principles and practice* by Rob J. Hyndman and George Athanasopoulos.
The online version of the text can be found [here](https://otexts.com/fpp3/).

## Install

### From Git
```shell
pip install git+https://github.com/xtreamsrl/pytsviz.git
```

### From PyPI
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
