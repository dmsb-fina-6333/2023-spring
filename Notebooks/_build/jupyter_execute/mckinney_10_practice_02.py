#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Practice (Wednesday 2:45 PM, Section 2)

# ## Announcements

# ## Practice

# ### Calulate the mean and standard deviation of returns by ticker for the MATANA (MSFT, AAPL, TSLA, AMZN, NVDA, and GOOG) stocks.

# Consider only dates with complete returns data.
# This calculation might be easier with a long data frame.

# ### Calculate the mean and standard deviation of returns and the maximum of closing prices by ticker for the MATANA stocks.

# This calculation might be easier if we pass a dictionary where the keys are the column names and the values are lists of functions.

# ### Calculate monthly means and volatilities for SPY and GOOG returns

# ### Plot the monthly means and volatilities from the previous exercise

# ### Calculate the *mean* monthly correlation between the Dow Jones stocks

# Drop duplicate correlations and self correlations (i.e., correlation between AAPL and AAPL), which are 1, by definition.

# ### Do correlations "go to one" during crises?

# How might we identify a "crisis"?

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
