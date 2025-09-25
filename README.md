# ML4H Project
This repository contains my approach to an exercise posed in the Machine 
Learning for Health Care course at ETHZ in 2025. This is ongoing work 
and intended for exploratory purposes only.

## The data and the goals
We are given observations of multivariate time series (of, say, $$p$$ features)
corresponding to measurements of patient data over a 24 hour ICU stay together 
with some outcomes, e.g., death vs. survival. The task in the exercise was to 
train a model to predict the outcomes from the observed time series. The 
challenges involved basic transformations (e.g., hourly bucketing), imputation,
and model specifications. We will focus on the following four tasks.
- Data preparation, exploration, and preliminary interpretation. There are
  multiple complications, e.g., several time series have very small variation
  per hour and most of the measurements aren't missing at random.
- As required in the exercise, we train some standard models for the prediction
  from hourly discretized data. Among others, we'll compare 
  XGBoost&mdash;somewhat the state-of-the-art for tabular data&mdash;, LSTM, and
  various simpler or more complicated models.
- We view the time series as an element of $$\mathrm{L}^2[0,24]^p$$ with very
  irregular measurements (component-wise missingness). We use a functional data
  analysis approach based on[^1] to separate the conditional distribution of
  time-series viewed as random trajectories according to the ourcome. The 
  methods aren't directly applicable because of the irregular discretization. 
  Hence we have to carefully handle missingness for discretized curves.

# Requirements
It is assumed that the data has been downloaded in .zip format from 
[https://physionet.org/content/challenge-2012/1.0.0/](https://physionet.org/content/challenge-2012/1.0.0/) 
and extracted into the root directory of the repository. We also assume 
that ``set-c.tar.gz`` has been extracted.

# The use of Jupytext
Everything is set up for both interactive use via jupyter notebooks and
for classical command line interaction. Jupytext is used to synchronize
jupyter notebooks and the corresponding .py files. The configuration is
contained in jupytext.toml and submitted to the repository for ease of
use.

# References
[^1] N. Zozoulenko, Th. Cass, and L. Gonon. Infinite-dimensional Mahalanobis
Distance with Applications to Kernelized Novelty Detection (2025), [https://arxiv.org/abs/2407.11873](arXiv).

