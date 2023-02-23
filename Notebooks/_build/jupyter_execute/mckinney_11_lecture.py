#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 11 - Time Series

# ## Introduction

# Chapter 11 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) discusses time series and panel data, which is where pandas ***shines***.
# We will use these time series and panel tools every day for the rest of the course.
# 
# We will focus on:
# 
# 1. Slicing a data frame or series by date or date range
# 1. Using `.shift()` to create leads and lags of variables
# 1. Using `.resample()` to change the frequency of variables
# 1. Using `.rolling()` to aggregate data over rolling windows
# 
# ***Note:*** 
# Indented block quotes are from McKinney unless otherwise indicated. 
# The section numbers here differ from McKinney because we will only discuss some topics.

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


# McKinney provides an excellent introduction to the concept of time series and panel data:
# 
# > Time series data is an important form of structured data in many different fields, such
# as finance, economics, ecology, neuroscience, and physics. Anything that is observed
# or measured at many points in time forms a time series. Many time series are fixed
# frequency, which is to say that data points occur at regular intervals according to some
# rule, such as every 15 seconds, every 5 minutes, or once per month. Time series can
# also be irregular without a fixed unit of time or offset between units. How you mark
# and refer to time series data depends on the application, and you may have one of the
# following:
# > - Timestamps, specific instants in time
# > - Fixed periods, such as the month January 2007 or the full year 2010
# > - Intervals of time, indicated by a start and end timestamp. Periods can be thought
# of as special cases of intervals
# > - Experiment or elapsed time; each timestamp is a measure of time relative to a
# particular start time (e.g., the diameter of a cookie baking each second since
# being placed in the oven)
# >
# >In this chapter, I am mainly concerned with time series in the first three categories,
# though many of the techniques can be applied to experimental time series where the
# index may be an integer or floating-point number indicating elapsed time from the
# start of the experiment. The simplest and most widely used kind of time series are
# those indexed by timestamp.
# 323
# >
# > pandas provides many built-in time series tools and data algorithms. You can effi‐
# ciently work with very large time series and easily slice and dice, aggregate, and
# resample irregular- and fixed-frequency time series. Some of these tools are especially
# useful for financial and economics applications, but you could certainly use them to
# analyze server log data, too.

# ## Time Series Basics

# Let us create a time series to play with.

# In[4]:


from datetime import datetime
dates = [
    datetime(2011, 1, 2), 
    datetime(2011, 1, 5),
    datetime(2011, 1, 7), 
    datetime(2011, 1, 8),
    datetime(2011, 1, 10), 
    datetime(2011, 1, 12)
]
np.random.seed(42)
ts = pd.Series(np.random.randn(6), index=dates)

ts


# Note that pandas converts the `datetime` objects to a pandas `DatetimeIndex` object and a single index value is a `Timestamp` object.

# In[5]:


ts.index


# In[6]:


ts.index[0]


# Recall that arithmetic operations between pandas objects automatically align on indexes.

# In[7]:


ts[::2]


# In[8]:


ts + ts[::2]


# ### Indexing, Selection, Subsetting

# We can use date and time labels to select data.

# In[9]:


stamp = ts.index[2]

stamp


# In[10]:


ts[stamp]


# pandas uses unambiguous date strings to select data.

# In[11]:


ts['1/10/2011'] # M/D/YYYY


# In[12]:


ts['20110110'] # YYYYMMDD


# In[13]:


ts['2011-01-10'] # YYYY-MM-DD


# In[14]:


ts['10-Jan-2011'] # D-Mon-YYYY


# In[15]:


ts['Jan-10-2011'] # Mon-D-YYYY


# But do not use U.K.-style dates.

# In[16]:


# ts['10/1/2011'] # D/M/YYYY


# Here is a longer time series for longer slices.

# In[17]:


np.random.seed(42)
longer_ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

longer_ts


# We can pass a year-month to slice all of the observations in May of 2001.

# In[18]:


longer_ts['2001-05']


# We can also pass a year to slice all observations in 2001.

# In[19]:


longer_ts['2001']


# If we sort our data chronologically, we can also slice with a range of date strings.

# In[20]:


ts['1/6/2011':'1/10/2011']


# To use date slices, our data should be sorted by the date index, as above.
# The following code works as though our data were sorted, but raises a warning that it will not work in future versions.

# In[21]:


ts2 = ts.sort_values()

ts2


# The following behavior is "deprecated", meaning it will eventually go away and we should not rely up on it.

# In[22]:


ts2['1/6/2011':'1/11/2011']


# In[23]:


ts2.sort_index()['1/6/2011':'1/11/2011']


# ***To be clear, a range of date strings is inclusive on both ends.***

# In[24]:


longer_ts['1/6/2001':'1/11/2001']


# ***Recall, if we modify a slice, we modify the original series or dataframe.***
# 
# > Remember that slicing in this manner produces views on the source time series like slicing NumPy arrays. This means that no data is copied and modifications on the slice will be reflected in the original data.

# ### Time Series with Duplicate Indices

# Most data in this course will be well-formed with one observation per datetime for series or one observation per individual per datetime for dataframes.
# However, you may later receive poorly-formed data with duplicate observations.
# The toy data in series `dup_ts` has three observations on February 2nd.

# In[25]:


dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)

dup_ts


# The `.is_unique` property tells us if an index is unique.

# In[26]:


dup_ts.index.is_unique


# In[27]:


dup_ts['1/3/2000']  # not duplicated


# In[28]:


dup_ts['1/2/2000']  # duplicated


# The solution to duplicate data depends on the context.
# For example, we may want the mean of all observations on a given date.
# The `.groupby()`  method can help us here.

# In[29]:


grouped = dup_ts.groupby(level=0)


# In[30]:


grouped.mean()


# In[31]:


grouped.last()


# Or we may want the number of observations on each date.

# In[32]:


grouped.count()


# ## Date Ranges, Frequencies, and Shifting

# > Generic time series in pandas are assumed to be irregular; that is, they have no fixed frequency. For many applications this is sufficient. However, it’s often desirable to work relative to a fixed frequency, such as daily, monthly, or every 15 minutes, even if that means introducing missing values into a time series. Fortunately pandas has a full suite of standard time series frequencies and tools for resampling, inferring frequencies, and generating fixed-frequency date ranges.
# 
# We will skip the sections on creating date ranges or different frequencies so we can focus on shifting data.

# ### Shifting (Leading and Lagging) Data

# ***Shifting is an important feature!***
# Shifting is moving data backward (or forward) through time.

# In[33]:


np.random.seed(42)
ts = pd.Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))

ts


# If we pass a positive integer $N$ to the `.shift()` method:
# 
# 1. The date index remains the same
# 2. Values are shifted down $N$ observations
# 
# "Lag" might be a better name than "shift" since a postive 2 makes the value at any timestamp the value from 2 timestamps above (earlier, since most time-series data are chronological).

# In[34]:


ts.shift() # if we do not specify "periods", pandas assumes 1


# In[35]:


ts.shift(2)


# If we pass a _negative_ integer $N$ to the `.shift()` method, values are shifted _up_ $N$ observations.

# In[36]:


ts.shift(-2)


# We will almost never shift with negative values (i.e., we will almost never bring forward values from the future) to prevent look-ahead bias.
# We do not want to assume that financial market participants have access to future data.
# Our most common shift will be to compute the percent change from one period to the next.
# We can calculate the percent change two ways.

# In[37]:


ts.pct_change()


# In[38]:


(ts - ts.shift()) / ts.shift()


# In[39]:


np.allclose(
    a=ts.pct_change(),
    b=(ts - ts.shift()) / ts.shift(),
    equal_nan=True
)


# Two observations on the percent change calculations above:
# 
# 1. The first percent change is NaN (missing) because there is no previous value to change from
# 2. The default `periods` argument for `.shift()`  and `.pct_change()` is 1
# 
# The naive shift examples above shift by a number of observations, without considering timestamps or their frequencies.
# As a result, timestamps are unchanged and values shift down (positive `periods` argument) or up (negative `periods` argument).
# However, we can also pass the `freq` argument to respect the timestamps.
# With the `freq` argument, timestamps shift by a multiple (specified by the `periods` argument) of datetime intervals (specified by the `freq` argument).
# Note that the examples below generate new datetime indexes.

# In[40]:


ts


# In[41]:


ts.shift(2, freq='M')


# In[42]:


ts.shift(3, freq='D')


# `M` is already months, so `T` is minutes.

# In[43]:


ts.shift(1, freq='90T')


# ### Shifting dates with offsets

# We can also shift timestamps to the beginning or end of a period or interval.

# In[44]:


from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()


# In[45]:


now + MonthEnd(0) # 0 is for move to the end of the month, but never leave the month


# In[46]:


now + MonthEnd(1) # 1 is for move to the end of the month, if already at end, move to the next end


# In[47]:


now + MonthEnd(2)


# Date offsets can help us align data for presentation or merging.
# ***But, be careful!***
# The default argument is 1, but we typically want 0.

# In[48]:


datetime(2021, 10, 30) + MonthEnd(0)


# In[49]:


datetime(2021, 10, 30) + MonthEnd(1)


# In[50]:


datetime(2021, 10, 31) + MonthEnd(0)


# In[51]:


datetime(2021, 10, 31) + MonthEnd(1)


# ## Resampling and Frequency Conversion

# ***Resampling is an important feature!***
# 
# > Resampling refers to the process of converting a time series from one frequency to
# another. Aggregating higher frequency data to lower frequency is called 
# downsampling, while converting lower frequency to higher frequency is called upsampling. Not
# all resampling falls into either of these categories; for example, converting W-WED
# (weekly on Wednesday) to W-FRI is neither upsampling nor downsampling.
# 
# We can resample both series and data frames.
# The `.resample()` method syntax is similar to the `.groupby()` method syntax.
# This similarity is because `.resample()` is syntactic sugar for `.groupby()`.

# ### Downsampling

# > Aggregating data to a regular, lower frequency is a pretty normal time series task. The
# data you’re aggregating doesn’t need to be fixed frequently; the desired frequency
# defines bin edges that are used to slice the time series into pieces to aggregate. For
# example, to convert to monthly, 'M' or 'BM', you need to chop up the data into 
# one-month intervals. Each interval is said to be half-open; a data point can only belong to
# one interval, and the union of the intervals must make up the whole time frame.
# There are a couple things to think about when using resample to downsample data:
# >
# > - Which side of each interval is closed
# > - How to label each aggregated bin, either with the start of the interval or the end

# In[52]:


rng = pd.date_range('2000-01-01', periods=12, freq='T')
ts = pd.Series(np.arange(12), index=rng)

ts


# We can aggregate the one-minute frequency data above to a five-minute frequency.
# Resampling requires and aggregation method, and here McKinney chooses the `.sum()` method.

# In[53]:


ts.resample('5min').sum()


# Two observations about the previous resampling example:
# 
# 1. For minute-frequency resampling, the default is that the new data are labeled by the left edge of the resampling interval
# 2. For minute-frequency resampling, the default is that the left edge is closed (included) and the right edge is open (excluded)
# 
# As a result, the first value of 10 at midnight is the sum of values at midnight and to the right of midnight, not including the value at 00:05 (i.e., $10 = 0+1+2+3+4$ at 00:00 and $35 = 5+6+7+8+9$  at 00:05).
# We can use the `closed` and `label` arguments to change this behavior.
# 
# In finance, we prefer `closed='right'` and `label='right'`.

# In[54]:


ts.resample('5min', closed='right', label='right').sum() 


# Mixed combinations of `closed` and `label` are possible but confusing.

# In[55]:


ts.resample('5min', closed='right', label='left').sum() 


# These defaults for minute-frequency data may seem odd, but any choice is arbitrary.
# I suggest you do the following when you use the `.resample()` method:
# 
# 1. Read the docstring
# 2. Check your output
# 
# pandas (and the `.resample()` method) are mature and widely used, so the defaults are typically reasonable.

# ### Upsampling and Interpolation

# To downsample (i.e., resample from higher frequency to lower frequency), we have to choose an aggregation method (e.g., `.mean()`, `.sum()`, `.first()`, or `.last()`).
# To upsample (i.e., resample from lower frequency to higher frequency), we do not have to choose an aggregation method.

# In[56]:


np.random.seed(42)
frame = pd.DataFrame(np.random.randn(2, 4),
                     index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])

frame


# We can use the `.asfreq()` method to convert to the new frequency "as is".

# In[57]:


df_daily = frame.resample('D').asfreq()

df_daily


# We do not *have* to choose an aggregation (disaggregation?) method, but we may want to choose a method to fill in the missing values.

# In[58]:


frame.resample('D').ffill()


# In[59]:


frame.resample('D').ffill(limit=2)


# In[60]:


frame.resample('W-THU').ffill()


# ## Moving Window Functions

# ***Moving window (or rolling window) functions are one of the neatest features of pandas, and we will frequently use moving window functions.***
# We will use data similar, but not identical, to the book data.
# ***We must remove the timezone if we want to merge with Fama-French data.***

# In[61]:


df = (
    yf.Tickers(tickers=['AAPL', 'MSFT', 'SPY'], session=session)
    .history(period='max', auto_adjust=False, progress=False)
    .assign(Date = lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)

df.head()


# The `.rolling()` method is similar to the `.groupby()` and `.resample()` methods.
# The `.rolling()` method accepts a window-width and requires an aggregation method.
# The next example calculates and plots the 252-trading day moving average of AAPL's price alongside the daily price.

# In[62]:


aapl = df.loc['2012':, ('Adj Close', 'AAPL')]
aapl.plot(label='Observed')
aapl.rolling(252).mean().plot(label='252 Trading Day Mean') # min_periods defaults to 252
aapl.rolling('365D').mean().plot(label='365 Calendar Day Mean') # min_periods defaults to 1
aapl.resample('A').mean().plot(style='.', label='Calendar Year Mean')
plt.legend()
plt.ylabel('AAPL Adjusted Close ($)')
plt.title('Comparison of Rolling and Resampling Means')
plt.show()


# Two observations:
# 
# 1. If we pass the window-width as an integer, the window-width is based on the number of observations and ignores time stamps
# 2. If we pass the window-width as an integer, the `.rolling()` method requires that number of observations for all windows (i.e., note that the moving average starts 251 trading days after the first daily price
# 
# We can use the `min_periods` argument to allow incomplete windows.
# For integer window widths, `min_periods defaults` to the given integer window width.
# For string date offsets, `min_periods defaults` to `1`.

# ### Binary Moving Window Functions

# Binary moving window functions accept two inputs.
# The most common example is the rolling correlation between two returns series.

# In[63]:


returns = df['Adj Close'].pct_change()

returns.head()


# In[64]:


returns['AAPL'].rolling(126, min_periods=100).corr(returns['SPY']).plot()
plt.ylabel('Correlation(AAPL, SPY)')
plt.title('Rolling Correlation between AAPL and SPY\n (126-Day Window w/ 100-Day Minimum)')
plt.show()


# In[65]:


returns[['AAPL', 'MSFT']].rolling(126, min_periods=100).corr(returns['SPY']).plot()
plt.ylabel('Correlation with SPY')
plt.title('Rolling Correlation with SPY\n (126-Day Window w/ 100-Day Minimum)')
plt.show()


# ### User-Defined Moving Window Functions

# Finally, we can define our own moving window functions and use the `.apply()` method to apply them
# However, note that `.apply()` will be much slower than the the optimized moving window functions (e.g., `.mean()`, `.std()`, etc.).
# 
# McKinney provides an abstract example here, but we will discuss a simpler example that calculates rolling volatility.
# Also, calculating rolling volatility with the `.apply()` method provides us a chance to benchmark it against the optimized version.

# In[66]:


returns['AAPL'].rolling(252).apply(np.std).mul(np.sqrt(252) * 100).plot() # annualize and convert to percent
plt.ylabel('Volatility (%)')
plt.title('Rolling Volatility\n (252-Day Window w/ 252-Day Minimum)')
plt.show()


# Do not be afraid to use `.apply()`, but realize that `.apply()` is typically 1000-times slower than the pre-built method.

# In[67]:


get_ipython().run_line_magic('timeit', "returns['AAPL'].rolling(252).apply(np.std)")


# In[68]:


get_ipython().run_line_magic('timeit', "returns['AAPL'].rolling(252).std()")


# In[ ]:




