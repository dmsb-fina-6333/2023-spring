#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 1 - Web Data, Log and Simple Returns, and Portfolio Math
# 
# This notebook covers three topics:
# 
# 1. How to download web data with the yfinance, pandas-datareader, and requests-cache packages
# 1. How to calculate log and simple returns
# 1. How to calculate portfolio returns

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# ## Web Data

# We will typically use the yfinance and pandas-datarader packages (combined with the requests-cache package) to download data from the web.
# 
# - If you followed my instructions to install Anaconda on your computer, you have already installed these packages
# - If you use DataCamp Workspace or Binder, I have already installed these packages
# - If you use Notheastern's Open OnDemand, they are working on my request, and you will have to install these packages every login by running the following in a code cell: `%pip install yfinance pandas-datareader requests-cache`

# ### The yfinance Package

# The [yfinance package](https://github.com/ranaroussi/yfinance) provides "a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance."
# Other packages provide similar functionality, but yfinance is best.
# We will use the [requests-cache package](https://github.com/requests-cache/requests-cache) to cache our data downloads locally.
# This local cache lets us reduce the number of times we ask the Yahoo! Finance application programming interface (API).

# In[3]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after=1)


# We can download data for the MATANA stocks (Microsoft, Alphabet, Tesla, Amazon, Nvidia, and Apple).
# We can pass tickers as either a space-delimited string or a list of strings.

# In[4]:


tickers = yf.Tickers(tickers='MSFT GOOG TSLA AMZN NVDA AAPL', session=session)
histories = tickers.history(period='max', auto_adjust=False, progress=False)
histories.index = histories.index.tz_localize(None)
histories


# In[5]:


( # Python ignores line breaks and white space inside ()
    histories # start with MATANA data frame
    ['Adj Close'] # slice adjusted close columns
    .pct_change() # calculate simple returns
    .loc['2022'] # select 2022 returns
    .add(1) # add 1
    .cumprod() # compound cumulative returns
    .sub(1) # subtract 1
    .mul(100) # convert decimals to percent
    .plot() # plot
)
plt.ylabel('Year-to-Date Return (%)')
plt.title('Year-to-Date Returns for MATANA Stocks')
plt.show()


# ### The pandas-datareader package

# The [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html) package provides easy access to various data sources, including [the Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [the Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/).
# The pandas-datareader package also downloads Yahoo! Finance data, but the yfinance package has better documentation.
# We will use `pdr` as the abbreviated prefix for pandas-datareader.

# In[6]:


import pandas_datareader as pdr


# Here we download the daily benchmark factors from Ken French's Data Library.
# The `get_available_datasets()` function provides the exact names for all of Ken French's data sets.

# In[7]:


pdr.famafrench.get_available_datasets(session=session)[:5]


# For Fama and French data, pandas-datareader returns the most recent five years of data unless we specify a `start` date.
# French typically provides data back through the second half of 1926.
# pandas-datareader returns dictionaries of data frames, and the `'DESCR'` value describes these data frames.

# In[8]:


ff_all = pdr.DataReader(
    name='F-F_Research_Data_Factors_daily',
    data_source='famafrench',
    start='1900', 
    session=session
)


# In[9]:


print(ff_all['DESCR'])


# In[10]:


(
    ff_all[0]
    .div(100)
    .add(1)
    .cumprod()
    .sub(1)
    .mul(100)
    .plot()
)
plt.ylabel('Cumulative Return (%)')
plt.title('Cumulative Returns for the Daily Benchmark Factors (%)')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()


# ## Log and Simple Returns

# We will typically use *simple* returns, calculated as $R_{simple,t} = \frac{P_t + D_t - P_{t-1}}{P_{t-1}} = \frac{P_t + D_t}{P_{t-1}} - 1$.
# The simple return is the return that investors receive on invested dollars.
# We can calculate simple returns from Yahoo Finance data with the `.pct_change()` method on the adjusted close column (i.e., `Adj Close`), which adjusts for dividends and splits.
# The adjusted close column is a reverse-engineered close price (i.e., end-of-trading-day price) that incorporates dividends and splits, making simple return calculations easy.

# However, we may see *log* returns elsewhere, which are the (natural) log of one plus simple returns:
# 
# $R_{log,t} = \log(1 + R_{simple,t}) = \log\left(1 +  \frac{P_t + D_t}{P_{t-1}} - 1 \right) = \log\left(\frac{P_t + D_t}{P_{t-1}} \right) = \log(P_t + D_t) - \log(P_{t-1})$
# 
# Therefore, we calculate log returns as either the log of one plus simple returns or the difference of the logs of the adjusted close column.
# Log returns are also known as *continuously-compounded* returns.

# We will typically use *simple* returns instead of *log* returns.
# However, this section explains the differences between simple and log returns and where each is appropriate.

# ### Simple and Log Returns are Similar for Small Returns

# $\log(1 + x) \approx x$ for small values of $x$, so simple returns and log returns are similar for small returns.
# Returns are typically small at daily and monthly horizons, so the difference between simple and log returns is small at these horizons.
# The following figure shows $R_{simple,t} \approx R_{log,t}$ for small $R$s.

# In[11]:


R = np.linspace(-0.75, 0.75, 100)
logR = np.log(1 + R)


# In[12]:


plt.plot(R, logR)
plt.plot([-1, 1], [-1, 1])
plt.xlabel('Simple Return')
plt.ylabel('Log Return')
plt.title('Log Versus Simple Returns')
plt.legend(['Actual', 'If Log = Simple'])
plt.show()


# ### Simple Return Advantage: Portfolio Calculations

# We can only perform portfolio calculations with simple returns.
# For a portfolio of $N$ assets with portfolio weights $w_i$, the portfolio return $R_{p}$ is the weighted average of the returns of its assets, $R_{p} = \sum_{i=1}^N w_i R_{i}$.
# For two stocks with portfolio weights of 50%, our portfolio return is $R_{portfolio} = 0.5 R_1 + 0.5 R_2 = \frac{R_1 + R_2}{2}$.
# However, we cannot calculate portfolio returns with log returns because the sum of logs is the log of products.
# 
# ***We cannot calculate portfolio returns as the weighted average of log returns.***

# ### Log Return Advantage: Log Returns are Additive

# The advantage of log returns is that we can compound log returns with addition.
# The additive property of log returns makes code simple, computations fast, and proofs easy when we compound returns over multiple periods.

# We compound returns from $t=0$ to $t=T$ as follows:
# 
# $1 + R_{0, T} = (1 + R_1) \times (1 + R_2) \times \dots \times (1 + R_T)$
# 
# Next, we take the log of both sides of the previous equation and use the property that the log of products is the sum of logs:
# 
# $\log(1 + R_{0, T}) = \log((1 + R_1) \times (1 + R_2) \times \dots \times (1 + R_T)) = \log(1 + R_1) + \log(1 + R_2) + \dots + \log(1 + R_T) = \sum_{t=1}^T \log(1 + R_t)$
# 
# Next, we exponentiate both sides of the previous equation:
# 
# $e^{\log(1 + R_{0, T})} = e^{\sum_{t=0}^T \log(1 + R_t)}$
# 
# Next, we use the property that $e^{\log(x)} = x$ to simplify the previous equation:
# 
# $1 + R_{0,T} = e^{\sum_{t=0}^T \log(1 + R_t)}$
# 
# Finally, we subtract 1 from both sides:
# 
# $R_{0 ,T} = e^{\sum_{t=0}^T \log(1 + R_t)} - 1$
# 
# So, the return $R_{0,T}$ from $t=0$ to $t=T$ is the exponentiated sum of log returns.
# The pandas developers assume users understand the math above and focus on optimizing sums.

# The following code generates 10,000 random log returns.
# The `np.random.randn()` call generates normally distributed random numbers.
# To generate equivalent simple returns, we exponentiate these log returns, then subtract one.

# In[13]:


np.random.seed(42)
df = pd.DataFrame(data={'R': np.exp(np.random.randn(10000)) - 1})


# In[14]:


df.describe()


# We can time the calculation of 12-observation rolling returns.
# We use `.apply()` for the simple return version because `.rolling()` does not have a product method.
# We find that `.rolling()` is slower with `.apply()` than with `.sum()` by a factor of 2,000.
# ***We will learn about `.rolling()` and `.apply()` in a few weeks, but they provide the best example of when to use log returns.***
# 

# In[15]:


get_ipython().run_cell_magic('timeit', '', "df['R12_via_simple'] = (\n    df['R']\n    .add(1)\n    .rolling(12)\n    .apply(lambda x: x.prod())\n    .sub(1)\n)")


# In[16]:


get_ipython().run_cell_magic('timeit', '', "df['R12_via_log'] = (\n    df['R']\n    .add(1)\n    .pipe(np.log)\n    .rolling(12)\n    .sum()\n    .pipe(np.exp)\n    .sub(1)\n)")


# In[17]:


np.allclose(df['R12_via_simple'], df['R12_via_log'], equal_nan=True)


# These two approaches calculate the same return, but the simple-return approach is 1,000 times slower than the log-return approach!

# ***We can use log returns to calculate total or holding period returns very quickly!***

# ## Portfolio Math

# Portfolio return $R_{p}$ is the weighted average of its asset returns, so $R_{p} = \sum_{i=1}^N w_i R_{i}$.
# Here $N$ is the number of assets, and $w_i$ is the weight on asset $i$.

# ### The 1/N Portfolio

# The $\frac{1}{N}$ portfolio equally weights portfolio assets, so $w_1 = w_2 = \dots = w_N = \frac{1}{N}$.
# We typically rebalance the $\frac{1}{N}$ portfolio every period.
# If $w_i = \frac{1}{N}$, then $R_{p} = \sum_{i=1}^N \frac{1}{N} R_{i} = \frac{\sum_{i=1}^N R_i}{N} = \bar{R}$.
# Therefore, we can use `.mean()` to calculate $\frac{1}{N}$ portfolio returns.

# In[18]:


returns = histories['Adj Close'].pct_change().loc['2022']


# In[19]:


returns


# In[20]:


returns.mean()


# In[21]:


rp_1 = returns.mean(axis=1)
rp_1


# ***Note that when we apply the same portfolio weights every period, we rebalance at the same frequency as the returns data.***
# If we have daily data, rebalance daily.
# If we have monthly data, we rebalance monthly, and so on.

# ### A More General Solution

# If we combine weights into vector $w$ and the time series of asset returns into matrix $\bf{R}$, then we can calculate the time series of portfolio returns as $R_p = w^T \bf{R}$.
# The pandas version of this calculation is `R.dot(w)`, where `R` is a data frame of asset returns and `w` is a series of portfolio weights.
# We can use this approach to calculate $\frac{1}{N}$ portfolio returns, too.

# In[22]:


weights = np.ones(returns.shape[1]) / returns.shape[1]
weights


# In[23]:


rp_2 = returns.dot(weights)
rp_2


# Both approaches give the same answer!

# In[24]:


np.allclose(rp_1, rp_2, equal_nan=True)


# In[ ]:




