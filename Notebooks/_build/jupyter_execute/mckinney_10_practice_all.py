#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Practice (All Sections)

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

ind.head()


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


# Here is the `.groupby()` solution!

# In[6]:


(
    ind
    .loc['2015':, ['Close']]
    .reset_index('Index')
    .groupby([pd.Grouper(freq='A'), 'Index'])
    .agg(['min', 'max'])
    ['Close']
    .unstack()
)


# ### Calulate the mean and standard deviation of returns by ticker for the MATANA (MSFT, AAPL, TSLA, AMZN, NVDA, and GOOG) stocks

# Consider only dates with complete returns data.
# Try this calculation with wide and long data frames, and confirm your results are the same.

# In[7]:


matana = (
    yf.Tickers(tickers='MSFT AAPL TSLA AMZN NVDA GOOG', session=session)
    .history(period='max', auto_adjust=False, progress=False)
    .rename_axis(columns=['Variable', 'Ticker'])
)


# In[8]:


returns = matana['Adj Close'].pct_change().dropna()


# In[9]:


returns.agg(['mean', 'std'])


# In[10]:


returns.stack().groupby('Ticker').agg(['mean', 'std']).T


# In[11]:


np.allclose(
    returns.agg(['mean', 'std']),
    returns.stack().groupby('Ticker').agg(['mean', 'std']).T    
)


# ### Calculate the mean and standard deviation of returns and the maximum of closing prices by ticker for the MATANA stocks

# Again, consider only dates with complete returns data.
# Try this calculation with wide and long data frames, and confirm your results are the same.

# In[12]:


_ = pd.MultiIndex.from_product([['Returns'], matana['Adj Close']])
matana[_] = matana['Adj Close'].pct_change()


# In[13]:


(
    matana
    .loc[returns.index]
    .stack()
    .groupby('Ticker')
    .agg({'Returns': ['mean', 'std'], 'Close': ['max']})
)


# ### Calculate monthly means and volatilities for SPY and GOOG returns

# In[14]:


spy_goog = (
    yf.Tickers(tickers='SPY GOOG', session=session)
    .history(period='max', auto_adjust=False, progress=False)
    .rename_axis(columns=['Variable', 'Ticker'])
)

spy_goog.head()


# In[15]:


spy_goog_m = (
    spy_goog
    .loc['1993-02':'2023-01', 'Adj Close']
    .pct_change()
    .groupby(pd.Grouper(freq='M'))
    .agg(['mean', 'std'])
)

spy_goog_m


# ### Plot the monthly means and volatilities from the previous exercise

# Here is a first go!

# In[16]:


spy_goog_m.plot(subplots=True)


# We can use `plt.subplots()` to better organize these plots.
# Here `.plt.subplots()` creates a tuple of empty axes and assign it to `ax`.
# Then we can use the `ax=` argument to assign each plot to each axes.
# I suggest ou pick up these tricks as you go instead of trying to read and memorize the matplotlib manual before you have a very specific task as hand.

# In[17]:


fig, ax = plt.subplots(2, 1, sharex=True)
spy_goog_m.swaplevel(axis=1)['mean'].mul(100).plot(ax=ax[0], ylabel='Mean of Daily Returns (%)')
spy_goog_m.swaplevel(axis=1)['std'].mul(100).plot(ax=ax[1], ylabel='SD of Daily Returns (%)')
plt.suptitle('Means and Standard Deviations of Daily Returns')
plt.show()


# ### Assign the Dow Jones stocks to five portfolios based on their monthly volatility

# First, we need to download Dow Jones stock data and calculate daily returns.
# Use data from 2020 through today.

# In[18]:


wiki = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')


# In[19]:


dj = (
    yf.Tickers(tickers=wiki[1]['Symbol'].to_list(), session=session)
    .history(period='max', auto_adjust=False, progress=False)
    .rename_axis(columns=['Variable', 'Ticker'])
)


# Here I add daily returns.

# In[20]:


_ = pd.MultiIndex.from_product([['Returns'], dj['Adj Close']])
dj[_] = dj['Adj Close'].pct_change()


# Here I add the monthly volatility to all the days in each month for each ticker.

# In[21]:


_ = pd.MultiIndex.from_product([['Volatility'], dj['Returns']])
dj[_] = dj['Returns'].groupby(pd.Grouper(freq='M')).transform('std')


# Here I assign stocks to portfolios based on their volatility.

# In[22]:


1 + pd.qcut(np.arange(10), q=5, labels=False)


# In[23]:


_ = pd.MultiIndex.from_product([['Portfolio'], dj['Volatility']])
dj[_] = dj['Volatility'].apply(pd.qcut, q=5, labels=False, axis=1)


# ### Plot the time-series volatilities of these five portfolios

# How do these portfolio volatilies compare to (1) each other and (2) the mean volatility of their constituent stocks?

# In[24]:


(
    dj
    .stack()
    .groupby(['Date', 'Portfolio'])
    ['Returns']
    .mean()
    .reset_index('Portfolio')
    .groupby([pd.Grouper(freq='M'), 'Portfolio'])
    .std()
    .rename(columns={'Returns': 'Volatility'})
    .reset_index('Portfolio')
    .dropna() # drop missing port.
    .assign(Portfolio = lambda x: 1 + x['Portfolio'].astype(int)) # w/o missing portf., can convert to integer
    .set_index('Portfolio', append=True)
    .unstack()
    ['Volatility']
    .mul(100)
    .plot()
)

plt.ylabel('Volatility of Daily Returns (%)')
plt.title('Volatility of Five Portfolios Formed on Volatility')
plt.show()


# ### Calculate the *mean* monthly correlation between the Dow Jones stocks

# Drop duplicate correlations and self correlations (i.e., correlation between AAPL and AAPL), which are 1, by definition.

# In[25]:


def mu_rho(x):
    rhos = x.corr()
    tril = np.tril(rhos)
    return np.nanmean(rhos.values[(tril != 0.) & (tril != 1.)])


# In[26]:


(
    dj
    ['Returns']
    .groupby(pd.Grouper(freq='Q'))
    .apply(mu_rho)
    .plot()
)

plt.ylabel('Mean Correlation')
plt.title('Mean Correlations of Dow-Jones Stocks')
plt.show()


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

# In[27]:


import pandas_datareader as pdr


# In[28]:


pdr.famafrench.get_available_datasets()[:5]


# In[29]:


ff_all = pdr.DataReader(
    name='F-F_Research_Data_Factors_daily',
    data_source='famafrench',
    start='1900',
    session=session
)


# In[30]:


(
    ff_all[0]
    .assign(Mkt = lambda x: x['Mkt-RF'] + x['RF'])
    ['Mkt']
    .groupby(pd.Grouper(freq='M'))
    .std()
    .mul(np.sqrt(252))
    .plot()
)

# adds vertical bands for U.S. wars
plt.axvspan('1941-12', '1945-09', alpha=0.25)
plt.annotate('WWII', ('1941-12', 90))
plt.axvspan('1950', '1953', alpha=0.25)
plt.annotate('Korean', ('1950', 80))
plt.axvspan('1959', '1975', alpha=0.25)
plt.annotate('Vietnam', ('1959', 90))
plt.axvspan('1990', '1991', alpha=0.25)
plt.annotate('Gulf I', ('1990', 80))
plt.axvspan('2001', '2021', alpha=0.25)
plt.annotate('Afghanistan', ('2001', 90))

plt.ylabel('Annualized Volatility of Daily Returns (%)')
plt.title('Volatility of U.S. Market')
plt.show()

