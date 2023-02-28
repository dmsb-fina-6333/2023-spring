#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 11 - Practice (Blank)

# ## Announcements

# ##  Practice

# ### Which are larger, overnight or intraday returns?

# Yahoo! Finance provides easy acces to high-quality open, high, low, close (OHLC) and adjusted close price data.
# However, Yahoo! Finance does not provide overnight or instraday returns directly.
# Therefore, we need to use math to decompose daily returns into overnight and intraday returns.
# 
# Daily returns are defined as (adjusted) closing price to (adjusted) closing price returns.
# Therefore, daily returns consist of overnight returns compounded with the intraday returns from the next day $(1 + R_{daily}) = (1 + R_{overnight}) \times (1 + R_{intraday})$ which we can rearrange to calculate overnight returns as $\frac{1 + R_{daily}}{1 + R_{intraday}} - 1 = R_{overnight}$.
# 
# We can calculate daily and intraday returns from Yahoo! Finance data as $R_{daily} = \frac{Adj\ Close_{t} - Adj\ Close_{t-1}}{Adj\ Close_{t-1}}$ and $R_{intraday} = \frac{Close - Open}{Open}$.
# 
# Compare the following for the SPY ETF:
# 
# 1. Cumulative returns with all available data
# 1. Total returns for each calendar year
# 1. Total returns over rolling 252-trading-day windows
# 1. Total returns over rolling 12-months windows after calculating monthly returns
# 1. Sharpe Ratios for each calendar year

# #### Cumulative returns with all available data

# #### Total returns for each calendar year

# #### Total returns over rolling 252-trading-day windows

# #### Total returns over rolling 12-months windows after calculating monthly returns

# #### Sharpe Ratios for each calendar year

# ### Calculate rolling betas
# 
# Calculate rolling capital asset pricing model (CAPM) betas for the MATANA stocks.
# 
# The CAPM says the risk premium on a stock depends on the risk-free rate, beta, and the risk premium on the market: $E(R_{stock}) = R_f + \beta_{stock} \times (E(R_{market}) - R_f)$.
# We can calculate CAPM betas as: $\beta_{stock} = \frac{Cov(R_{stock} - R_f, R_{market} - R_f)}{Var(R_{market} - R_f)}$.

# ### Calculate rolling Sharpe Ratios

# Calculate rolling Sharpe Ratios for the MATANA stocks.
# 
# The Sharpe Ratio is often used to evaluate fund managers.
# The Sharpe Ratio is $SR_i = \frac{\overline{R_i - R_f}}{\sigma}$, where $\overline{R_i-R_f}$ is mean fund return relative to the risk-free rate over some period and $\sigma$ is the standard deviation of $R_i-R_f$ over the same period.
# While the Sharpe Ratio is typically used for funds, we can apply it to a single stock to test our knowledge of the `.rolling()` method.
# Calculate and plot the one-year rolling Sharpe Ratio for the MATANA stocks using all available daily data.

# ### Does more frequent rebalancing increase or decrease returns?

# Compare decade-total returns for the following rebalancing frequencies:
# 
# 1. Daily rebalancing
# 1. Monthly rebalancing
# 1. Annual rebalancing
# 1. Decade rebalancing
# 
# Use equally-weighted portfolios of industry-level daily returns from French's website: `'17_Industry_Portfolios_daily'`.

# In[ ]:




