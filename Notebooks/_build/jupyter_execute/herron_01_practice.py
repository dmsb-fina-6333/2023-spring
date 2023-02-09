#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 1 - Practice (Blank)

# ## Announcements

# ## Practice

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf
import pandas_datareader as pdr
import requests_cache


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format
session = requests_cache.CachedSession(expire_after=1)


# ### Download all available daily price data for tickers TSLA, F, AAPL, AMZN, and META to data frame `histories`

# Remove time zone information from the index and use `histories.columns.names` to label the variables and tickers as `Variable` and `Ticker`.

# ### Calculate all available daily returns and save to data frame `returns`

# ### Slices returns for the 2020s and assign to `returns_2020s`

# ### Download all available data for the Fama and French daily benchmark factors to dictionary `ff_all`

# I often use the following code snippet to find the exact name for the the daily benchmark factors file.

# In[3]:


pdr.famafrench.get_available_datasets()[:5]


# ### Slice the daily benchmark factors, convert them to decimal returns, and assign to `ff`

# ### Use the `.cumprod()` method to plot cumulative returns for these stocks in the 2020s

# ### Use the `.cumsum()` method with log returns to plot cumulative returns for these stocks in the 2020s

# ### Use price data only to plot cumulative returns for these stocks in the 2020s

# ### Calculate the Sharpe Ratio for TSLA

# Calculate the Sharpe Ratio with all available returns and 2020s returns.
# Recall the Sharpe Ratio is $\frac{\overline{R_i - R_f}}{\sigma_i}$, where $\sigma_i$ is the volatility of *excess* returns.
# 
# ***I suggest you write a function named `sharpe()` to use for the rest of this notebook.***

# ### Calculate the market beta for TSLA

# Calculate the market beta with all available returns and 2020s returns.
# Recall we estimate market beta with the ordinary least squares (OLS) regression $R_i-R_f = \alpha + \beta (R_m-R_f) + \epsilon$.
# We can estimate market beta with the covariance formula above for a univariate regression if we do not need goodness of fit statistics.
# 
# ***I suggest you write a function named `beta()` to use for the rest of this notebook.***

# ### Guess the Sharpe Ratios for these stocks in the 2020s

# ### Guess the market betas for these stocks in the 2020s

# ### Calculate the Sharpe Ratios for these stocks in the 2020s

# How good were your guesses?

# ### Calculate the market betas for these stocks in the 2020s

# How good were your guesses?

# ### Calculate the Sharpe Ratio for an *equally weighted* portfolio of these stocks in the 2020s

# What do you notice?

# ### Calculate the market beta for an *equally weighted* portfolio of these stocks in the 2020s

# What do you notice?

# ### Calculate the market betas for these stocks every calendar year for every possible year

# Save these market betas to data frame `betas`.
# Our current Python knowledge limits us to a for-loop, but we will learn easier and faster approaches soon!

# ### Plot the time series of market betas
