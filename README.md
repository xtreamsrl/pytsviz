# pytsviz

*pytsviz* is a suite of tools to quickly analyze and visualize time series data. It is partially based on the [*tsviz*](https://github.com/xtreamsrl/tsviz) R package.

The *utils* module contains a set of useful utilies, not strictly realted to visualization, we often use (e.g. harmonics computation).
The *viz* module contains a lot of functions to quickly visualize the main aspects of a univariate time series. Most of them can be run backed by either *matplotlib* or *Plotly*.


## Install

```shell
pip install git+https://github.com/xtreamsrl/pytsviz.git
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

Please also make sure that function documntation is consistent. We are currently using [Sphinx docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).
