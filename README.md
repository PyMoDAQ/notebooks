# Notebooks


This package contains Various Jupyter notebooks to illustrate some PyMoDAQ features:

# About PyMoDAQ's Data


Data in PyMoDAQ are featured with a lot of metadata, allowing their
proper description and enabling seamlessly saving/loading to hdf5 files.
But what about representation? Analysis? Exploration?

## Plotting Data

With Python you usually have to write a script to manipulate and
plot your data using your favorite backend (matplotlib, plotly, qt,
tkinter, …) However because PyMoDAQ is highly graphical you won't need
that. PyMoDAQ is featured with various data viewers allowing you to plot
any kind of data. You'll see below some nice examples of how to plot
your PyMoDAQ's data using the builtin data viewers.

Notebook : [Plotting Data](https://github.com/PyMoDAQ/notebooks/blob/main/notebooks/plotting_data.ipynb)


## Data Analysis

PyMoDAQ's data are also advanced objects including a lot of properties and methods to explore and manipulate
them. A practical example of data analysis from a physics experiment is presented in the following notebook. You'll
see how to crop, average, fit... your data for an easier analysis.

Notebook : [Analyzing Data](https://github.com/PyMoDAQ/notebooks/blob/main/notebooks/data_analysis_tutorial.ipynb)


# Bayesian Optimisation and Gaussian Processes

The following notebook illustrates the use of Bayesian Optimisation of Black box functions using Gaussian Processes
and Utility functions. Such an algorithm is used in the PyMoDAQ extension: Bayesian Optimisation.

Notebook : [Bayesian Optimisation](https://github.com/PyMoDAQ/notebooks/blob/main/notebooks/gaussian_process.ipynb)
