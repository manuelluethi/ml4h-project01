# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Exploratory data analysis and preparations
# In this notebook, we perform some exploratory data analysis together with
# basic preliminary transformations. These include

# %%
# Import dependencies
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# %% tags=["active-ipynb"]
# # Important directories
# ROOT_DIR = os.path.dirname(os.getcwd())
# DATA_DIR = os.path.join(ROOT_DIR, 'data')
# LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# %% tags=["active-py"] jupyter={"source_hidden": true}
# Only in command line mode:
# os.getcwd() doesn't work if the script is run from the command line.
# This does the same as above for CLI and isn't needed nor does it work in the
# notebook.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# %%
# Setup logging
logger = logging.getLogger('data-exploration-logger')
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    # If the logger already has handlers, remove them
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

if not logger.hasHandlers():
    # Create a directory for logs if it doesn't exist
    try:
        os.mkdir(LOG_DIR)
    except FileExistsError:
        pass

    # Create a file handler
    log_file = os.path.join(LOG_DIR, 'data-exploration.log')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)

    # Create a formatter and set it for both handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s:\n\t\t%(message)s'
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


# Function to load the datasets
def loadData():
    names = ['a', 'b', 'c']
    dataSets = []
    for s in names:
        featuresPath = os.path.join(DATA_DIR, f'set-{s}.parquet.gzip')
        labelsPath = os.path.join(DATA_DIR, f'set-{s}-outcomes.parquet.gzip')
        dfFeatures = pd.read_parquet(featuresPath)
        dfLabels = pd.read_parquet(labelsPath)
        dataSets.append([dfFeatures, dfLabels])
    return dataSets


# we load the various data sets
dataSets = loadData()


# %%
# Preperations:
# A list of all the time series variables to consider
LIST_VARIABLE_TS = [
    'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'ICUType', 'K', 'Lactate', 'MAP', 'MechVent',
    'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
    'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
    'WBC', 'pH'
]

# A list of static variables
LIST_VARIABLE_STATIC = [
    'Age', 'Gender', 'Height', 'Weight'
]

# A list of keys:
# RecordID indexes the individual times series
# Hour is the time stamp to be used -- note that we will have to normalize
# the variable Time which is in format hh:mm after admission to hourly
# variables
LIST_VARIABLE_KEYS = [
    'RecordID', 'Hour'
]


# %% [markdown]
# ### Data exploration
# We visualize the time series data using boxplots.


# %%
def boxplotTimeSeries(ax, data, variable, title=None):
    'Create a sequence of boxplots for a given time series variable.'
    sns.set(style="whitegrid")
    df = data[['Hour', variable]].copy(deep=True)
    # We normalize the hourly data around its median for better display
    # Also, we only use the inner 95% around the median
    medians = df.groupby('Hour')[variable].transform('median')
    df[variable] = df[variable] - medians
    df['quantile_low'] = df.groupby('Hour')[variable].transform(
        'quantile', 0.025
    )
    df['quantile_high'] = df.groupby('Hour')[variable].transform(
        'quantile', 0.975
    )
    # We drop everything outside the 95% quantile range
    df = df[pd.isna(df[variable]) | (df[variable] >= df['quantile_low']) &
            (df[variable] <= df['quantile_high'])]
    plt = sns.boxplot(ax=ax, data=df, x='Hour', y=variable)
    plt.set_title(title
                  if title
                  else f'Distribution of {variable} around its median')
    plt.set_xlabel("")
    plt.set_ylabel("")
    # we symmetrically set the y-axis limits
    y_abs_max = np.max((np.abs(df[variable]).max(), 0.05))
    plt.set_ylim(-y_abs_max, y_abs_max)
    return plt


# %%
def boxPlotTimeSeriesMultiple(data, variables, title=None):
    'Create a stack of boxplots for multiple time series variables.'
    n = len(variables)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 3 * n))
    'for each variable, create a boxplot and add it to the stacked plot'
    for i, variable in enumerate(variables):
        plot = boxplotTimeSeries(axes[i], data, variable, f'{variable}')
        axes[i].tick_params(axis='x', labelbottom=False)
    fig.suptitle(title if title else
                 ('Boxplots of Time Series Variables'
                  'around the hourly median'),
                 fontsize=16)
    # Adjust vertical spacing of the plots and reduce the top due to supposedly
    # known issue that the suptitle is plotted above an empty first plot. I
    # couldn't extract this info from the matplotlib documentation.
    fig.subplots_adjust(hspace=0.2, top=1 - 1 / n)


# %% [markdown]
# ### Boxplots of time series variables
# We plot the boxplots of the time series variables around their hourly median
# for qualitative assessment of questions like homoskedasticity vs.
# heteroskedasticity and missingness. We note that some variables are measured
# very irregularly and that most variables are highly heteroskedastic.


# %%
title = ('Hourly distributions of time series variables around '
         'their hourly median')
# Number of variables to plot
boxPlotTimeSeriesMultiple(dataSets[0][0],
                          LIST_VARIABLE_TS,
                          title=title)


# ## [markdown]
# ### Missingness
# For better understanding of missingness, we plot a heatmap for missing data.


# %%
def missingHeatMap(ax, data, variable, title=None):
    sns.set(style="whitegrid")
    df = data[['Hour', variable]].copy(deep=True)
    df['missing'] = df[variable].isna().astype(int)
    grouped = df.groupby('Hour')['missing'].agg(['sum', 'count'])
    grouped['pct_missing'] = grouped['sum'] / grouped['count']
    heatmap_data = grouped.pivot_table(columns='Hour', values='pct_missing')
    plt = sns.heatmap(heatmap_data,
                      vmin=0, vmax=1,
                      cmap='Reds',
                      ax=ax,
                      cbar=True)
    plt.set_title(title
                  if title
                  else f'Distribution of {variable} around its median')
    plt.set_xlabel("")
    plt.set_ylabel("")
    return plt


# %%
def missingHeatMapMultiple(data, variables, title=None):
    'Create a stack of heatmaps for multiple time series variables.'
    n = len(variables)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2 * n))
    for i, variable in enumerate(variables):
        plot = missingHeatMap(axes[i], data, variable, f'{variable}')
        axes[i].tick_params(axis='x', labelbottom=False)
    fig.suptitle(title if title else
                 ('Heatmaps of Missingness of Time Series Variables'),
                 fontsize=16)
    fig.subplots_adjust(hspace=0.2, top=1 - 1 / n)


# %%
title = 'Heatmaps of Missingness of Time Series Variables'
missingHeatMapMultiple(dataSets[0][0], LIST_VARIABLE_TS, title=title)
