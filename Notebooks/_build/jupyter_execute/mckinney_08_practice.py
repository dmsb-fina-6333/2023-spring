#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 8 - Practice (Blank)

# ## Announcements

# ## Practice

# ### Download data from Yahoo! Finance for BAC, C, GS, JPM, MS, and PNC and assign to data frame `stocks`.

# Use `stocks.columns.names` to assign the names `Variable` and `Ticker` to the column multi index.

# ### Reshape `stocks` from wide to long with dates and tickers as row indexes and assign to data frame `stocks_long`.

# ### Add daily returns for each stock to data frames `stocks` and `stocks_long`.

# Name the returns variable `Returns`, and maintain all multi indexes.
# *Hint:* Use `pd.MultiIndex()` to create a multi index for the the wide data frame `stocks`.

# ### Download the daily benchmark return factors from Ken French's data library.

# ### Add the daily benchmark return factors to `stocks` and `stocks_long`.

# For the wide data frame `stocks`, use the outer index name `Factors`.

# ### Write a function `download()` that accepts tickers and returns a wide data frame of returns with the daily benchmark return factors.

# ### Download earnings per share for the stocks in `stocks` and combine to a long data frame `earnings`.

# Use the `.earnings_dates` method described [here](https://pypi.org/project/yfinance/).
# Use `pd.concat()` to combine the result of each the `.earnings_date` data frames and assign them to a new data frame `earnings`.
# Name the row indexes `Ticker` and `Date` and swap to match the order of the row index in `stocks_long`.

# In[1]:


# some students had to update yfinance to use the .earnings_dates atrtibute
# %pip install -U yfinance


# ### Combine `earnings` with the returns from `stocks_long`.

# ***It is easier to leave `stocks` and `stocks_long` as-is and work with slices `returns` and `returns_long`.***
# Use the `tz_localize('America/New_York')` method add time zone information back to `returns.index` and use `pd.to_timedelta(16, unit='h')` to set time to the market close in New York City.
# Use [`pd.merge_asof()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html) to match earnings announcement dates and times to appropriate return periods.
# For example, if a firm announces earnings after the close at 5 PM on February 7, we want to match the return period from 4 PM on February 7 to 4 PM on February 8.

# ### Plot the relation between daily returns and earnings surprises

# Three options in increasing difficulty:
# 
# 1. Scatter plot
# 1. Scatter plot with a best-fit line using `regplot()` from the seaborn package
# 1. Bar plot using `barplot()` from the seaborn package after using `pd.qcut()` to form five groups on earnings surprises

# ### Repeat the earnings exercise with the S&P 100 stocks

# ### Repeat the earnings exercise with *excess returns* of the S&P 100 Stocks

# Excess returns are returns minus market returns.
# We need to add a timezone and the closing time to the market return from Fama and French.

# ### Improve your `download()` function from above

# Modify `download()` to accept one or more than one ticker.
# Since we will not use the advanced functionality of the tickers object that `yf.Tickers()` creates, we will use `yf.download()`.
# The current version of `yf.download()` does not accept a `session=` argument.
