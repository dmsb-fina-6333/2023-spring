#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 8 - Practice (Monday 2:45 PM, Section 3)

# ## Announcements

# 1. Quiz 3 mean was $\approx 80\%$
# 1. I posted project 1 to Canvas

# ## Practice

# ### Download data from Yahoo! Finance for BAC, C, GS, JPM, MS, and PNC and assign to data frame `stocks`.

# Use `stocks.columns.names` to assign the names `Variable` and `Ticker` to the column multi index.

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
session = requests_cache.CachedSession(expire_after=1)


# In[4]:


tickers = yf.Tickers('BAC C GS JPM MS PNC', session=session)
stocks = tickers.history(period='max', auto_adjust=False)
stocks.index = stocks.index.tz_localize(None)
stocks.columns.names = ['Variable', 'Ticker']
stocks.head()


# ### Reshape `stocks` from wide to long with dates and tickers as row indexes and assign to data frame `stocks_long`.

# In[5]:


stocks_long = stocks.stack()
stocks_long.head()


# ### Add daily returns for each stock to data frames `stocks` and `stocks_long`.

# Name the returns variable `Returns`, and maintain all multi indexes.
# *Hint:* Use `pd.MultiIndex()` to create a multi index for the the wide data frame `stocks`.

# In[6]:


_ = pd.MultiIndex.from_product([['Returns'], stocks['Adj Close'].columns])
stocks[_] = stocks['Adj Close'].pct_change()
stocks.head()


# The easiest way to add returns to long data frame `stocks_long` is to `.stack()` wide data frame `stocks`!
# We could sort `stocks_long` by ticker and date (to sort chronologically within each ticker), then use `.pct_change()`.
# However, this approach miscalculates the first return for every ticker except for the first ticker.
# The easiest and safest solution is to `.stack()` the wide data frame `stocks`!

# In[7]:


# see that the first return for C is wrong
# stocks_long['Adj Close'].sort_index(level=['Ticker', 'Date']).pct_change().loc[(slice(None), 'C')]


# In[8]:


stocks_long = stocks.stack()


# ### Download the daily benchmark return factors from Ken French's data library.

# In[9]:


pdr.famafrench.get_available_datasets()[:5]


# In[10]:


ff = (
    pdr.DataReader(
        name='F-F_Research_Data_Factors_daily',
        data_source='famafrench',
        start='1900',
        session=session
    )
    [0]
    .div(100)
)


# ### Add the daily benchmark return factors to `stocks` and `stocks_long`.

# For the wide data frame `stocks`, use the outer index name `Factors`.

# In[11]:


_ = pd.MultiIndex.from_product([['Factors'], ff.columns])
stocks[_] = ff
stocks.head()


# We can use `.join()` even though `stocks_long` has a multi index.
# ***Note that re-running a "self join" can create duplicate columns.***
# We should be careful to run self joins only once!

# In[12]:


stocks_long = stocks_long.join(ff)
stocks_long.head()


# ### Write a function `download()` that accepts tickers and returns a wide data frame of returns with the daily benchmark return factors.

# In[13]:


def download(**kwargs):
    # get stocks data
    tickers = yf.Tickers(**kwargs)
    stocks = tickers.history(period='max', auto_adjust=False)
    stocks.index = stocks.index.tz_localize(None)
    stocks.columns.names = ['Variable', 'Ticker']

    _ = pd.MultiIndex.from_product([['Returns'], stocks['Adj Close'].columns])
    stocks[_] = stocks['Adj Close'].pct_change()

    # get factor data
    ff = (
        pdr.DataReader(
            name='F-F_Research_Data_Factors_daily',
            data_source='famafrench',
            start='1900',
            session=session
        )
        [0]
        .div(100)
    )

    # combine
    _ = pd.MultiIndex.from_product([['Factors'], ff.columns])
    stocks[_] = ff

    return stocks


# Below I will provide a more compact and flexible version of `download()`.

# ### Download earnings per share for the stocks in `stocks` and combine to a long data frame `earnings`.

# Use the `.earnings_dates` method described [here](https://pypi.org/project/yfinance/).
# Use `pd.concat()` to combine the result of each the `.earnings_date` data frames and assign them to a new data frame `earnings`.
# Name the row indexes `Ticker` and `Date` and swap to match the order of the row index in `stocks_long`.

# In[14]:


# some students had to update yfinance to use the .earnings_dates atrtibute
# %pip install -U yfinance


# In[15]:


earnings = (
    pd.concat(
        objs=[tickers.tickers[t].earnings_dates for t in tickers.tickers],
        keys=tickers.tickers,
        names=['Ticker', 'Date']
    )
    .swaplevel()
    .rename_axis(columns='Variable')
)

earnings.tail()


# ### Combine `earnings` with the returns from `stocks_long`.

# ***It is easier to leave `stocks` and `stocks_long` as-is and work with slices `returns` and `returns_long`.***
# Use the `tz_localize('America/New_York')` method add time zone information back to `returns.index` and use `pd.to_timedelta(16, unit='h')` to set time to the market close in New York City.
# Use [`pd.merge_asof()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html) to match earnings announcement dates and times to appropriate return periods.
# For example, if a firm announces earnings after the close at 5 PM on February 7, we want to match the return period from 4 PM on February 7 to 4 PM on February 8.

# In[16]:


returns = stocks['Returns']
returns.index = returns.index.tz_localize('America/New_York') + pd.to_timedelta(16, unit='h')
returns_long = returns.stack().to_frame('Returns')
returns_long.columns.name = 'Variable'
returns_long


# In[17]:


earnings_2 = (
    pd.merge_asof(
        left=earnings.sort_index(level=['Date', 'Ticker']),
        right=returns_long.sort_index(level=['Date', 'Ticker']),
        on='Date',
        by='Ticker',
        direction='forward',
        allow_exact_matches=False
    )
    .dropna()
    .reset_index()
    .set_index(['Date', 'Ticker'])
)

earnings_2.head()


# ### Plot the relation between daily returns and earnings surprises

# Three options in increasing difficulty:
# 
# 1. Scatter plot
# 1. Scatter plot with a best-fit line using `regplot()` from the seaborn package
# 1. Bar plot using `barplot()` from the seaborn package after using `pd.qcut()` to form five groups on earnings surprises

# In[18]:


(
    earnings_2
    [['Surprise(%)', 'Returns']]
    .mul(100)
    .plot(x='Surprise(%)', y='Returns', kind='scatter')
)
plt.xlabel('Earnings Suprise (%)')
plt.ylabel('Announcement Return (%)')

_ = ' '.join(earnings_2.index.get_level_values('Ticker').unique())
__ = earnings_2.index.get_level_values('Date')
plt.title(f'Earnings Announcements\n for {_}\n from {__.min():%B %Y} to {__.max():%B %Y}')
plt.show()


# In[19]:


import seaborn as sns


# In[20]:


sns.regplot(
    x='Surprise(%)',
    y= 'Returns',
    data=earnings_2[['Surprise(%)', 'Returns']].mul(100)
)

plt.xlabel('Earnings Suprise (%)')
plt.ylabel('Announcement Return (%)')

_ = ' '.join(earnings_2.index.get_level_values('Ticker').unique())
__ = earnings_2.index.get_level_values('Date')
plt.title(f'Earnings Announcements\n for {_}\n from {__.min():%B %Y} to {__.max():%B %Y}')
plt.show()


# In[21]:


earnings_2['ESQ'] = pd.qcut(x=earnings_2['Surprise(%)'], q=5, labels=False)


# In[22]:


sns.barplot(
    x='ESQ',
    y= 'Returns',
    data=(
        earnings_2
        [['Surprise(%)', 'Returns']]
        .mul(100)
        .assign(ESQ = lambda x: pd.qcut(x=x['Surprise(%)'], q=5, labels=False))
    )
)

plt.xlabel('Earnings Suprise Portfolio')
plt.ylabel('Announcement Return (%)')

_ = ' '.join(earnings_2.index.get_level_values('Ticker').unique())
__ = earnings_2.index.get_level_values('Date')
plt.title(f'Earnings Announcements\n for {_}\n from {__.min():%B %Y} to {__.max():%B %Y}')
plt.show()


# ***There is a positive relation between announcment returns and earnings surprises!***
# Of course, to say more we need more data and to control for market movements, but this analaysis is a start!
