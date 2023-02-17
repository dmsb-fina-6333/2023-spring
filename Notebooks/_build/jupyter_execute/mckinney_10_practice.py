#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Practice (Blank)

# ## Announcements

# ## Practice

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()


# ### Replicate the following `.pivot_table()` output with `.groupby()`

# In[4]:


ind = (
    yf.download(
        tickers='^GSPC ^DJI ^IXIC ^FTSE ^N225 ^HSI',
        progress=False
    )
    .rename_axis(columns=['Variable', 'Index'])
    .stack()
)


# In[5]:


(
    ind
    .loc['2015':]
    .reset_index()
    .pivot_table(
        values='Close',
        index=pd.Grouper(key='Date', freq='A'),
        columns='Index',
        aggfunc=['min', 'max']
    )
)


# ### Calulate the mean and standard deviation of returns by ticker for the MATANA (MSFT, AAPL, TSLA, AMZN, NVDA, and GOOG) stocks

# Consider only dates with complete returns data.
# Try this calculation with wide and long data frames, and confirm your results are the same.

# ### Calculate the mean and standard deviation of returns and the maximum of closing prices by ticker for the MATANA stocks

# Again, consider only dates with complete returns data.
# Try this calculation with wide and long data frames, and confirm your results are the same.

# ### Calculate monthly means and volatilities for SPY and GOOG returns

# ### Plot the monthly means and volatilities from the previous exercise

# ### Assign the Dow Jones stocks to five portfolios based on their monthly volatility

# First, we need to download Dow Jones stock data and calculate daily returns.
# Use data from 2019 through today.

# ### Plot the time-series volatilities of these five portfolios

# How do these portfolio volatilies compare to (1) each other and (2) the mean volatility of their constituent stocks?

# ### Calculate the *mean* monthly correlation between the Dow Jones stocks

# Drop duplicate correlations and self correlations (i.e., correlation between AAPL and AAPL), which are 1, by definition.

# ### Is market volatility higher during wars?

# Here is some guidance:
# 
# 1. Download the daily factor data from Ken French's website
# 1. Calculate daily market returns by summing the market risk premium and risk-free rates (`Mkt-RF` and `RF`, respectively)
# 1. Calculate the volatility (standard deviation) of daily returns *every month* by combining `pd.Grouper()` and `.groupby()`)
# 1. Multiply by $\sqrt{252}$ to annualize these volatilities of daily returns
# 1. Plot these annualized volatilities
# 
# Is market volatility higher during wars?
# Consider the following dates:
# 
# 1. WWII: December 1941 to September 1945
# 1. Korean War: 1950 to 1953
# 1. Viet Nam War: 1959 to 1975
# 1. Gulf War: 1990 to 1991
# 1. War in Afghanistan: 2001 to 2021
