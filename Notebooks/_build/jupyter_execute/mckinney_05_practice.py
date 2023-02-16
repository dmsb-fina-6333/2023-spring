#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 5 - Practice (Blank)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf
import requests_cache


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format
session = requests_cache.CachedSession()


# In[3]:


tickers = yf.Tickers('AAPL IBM MSFT GOOG', session=session)
prices = tickers.history(period='max', auto_adjust=False, progress=False)
prices.index = prices.index.tz_localize(None)
returns = prices['Adj Close'].pct_change().dropna()
returns


# ## Practice

# ### What are the mean daily returns for these four stocks?

# ### What are the standard deviations of daily returns for these four stocks?

# ### What are the *annualized* means and standard deviations of daily returns for these four stocks?

# ### Plot *annualized* means versus standard deviations of daily returns for these four stocks

# Use `plt.scatter()`, which expects arguments as `x` (standard deviations) then `y` (means).

# ### Repeat the previous calculations and plot for the stocks in the Dow-Jones Industrial Index (DJIA)

# We can find the current DJIA stocks on [Wikipedia](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average).
# We will need to download new data, into `tickers2`, `prices2`, and `returns2`.

# ### Calculate total returns for the stocks in the DJIA

# We can use the `.prod()` method to compound returns as $1 + R_T = \prod_{t=1}^T (1 + R_t)$.
# Technically, we should write $R_T$ as $R_{0,T}$, but we typically omit the subscript $0$.

# ### Plot the distribution of total returns for the stocks in the DJIA

# We can plot a histogram, using either the `plt.hist()` function or the `.plot(kind='hist')` method.

# ### Which stocks have the minimum and maximum total returns?

# ### Plot the cumulative returns for the stocks in the DJIA

# We can use the cumulative product method `.cumprod()` to calculate the right hand side of the formula above.

# ### Repeat the plot above with only the minimum and maximum total returns
