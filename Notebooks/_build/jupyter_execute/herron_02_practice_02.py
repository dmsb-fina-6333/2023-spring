#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 2 - Practice (Wednesday 2:45 PM, Section 2)

# ## Announcements

# ##  Practice

# ### Implement the SMA(20) strategy with Bitcoin from the lecture notebook

# Try to create the `btc` data frame in one code cell with one assignment (i.e., one `=`).

# ### How does SMA(20) outperform buy-and-hold with this sample?

# Consider the following:
# 
# 1. Does SMA(20) avoid the worst performing days? How many of the worst 20 days does SMA(20) avoid? Try the `.sort_values()` or `.nlargest()` method.
# 1. Does SMA(20) preferentially avoid low-return days? Try to combine the `.groupby()` method and `pd.qcut()` function.
# 1. Does SMA(20) preferentially avoid high-volatility days? Try to combine the `.groupby()` method and `pd.qcut()` function.

# ### Implement the SMA(20) strategy with the market factor from French

# We need to impute a market price before we calculate SMA(20).

# ### How often does SMA(20) outperform buy-and-hold with 10-year rolling windows?

# ### Implement a long-only BB(20, 2) strategy with Bitcoin

# More on Bollinger Bands [here](https://www.bollingerbands.com/bollinger-bands) and [here](https://www.bollingerbands.com/bollinger-band-rules).
# In short, Bollinger Bands are bands around a trend, typically defined in terms of simple moving averages and volatilities.
# Here, long-only BB(20, 2) implies we have upper and lower bands at 2 standard deviations above and below SMA(20):
# 
# 1. Buy when the closing price crosses LB(20) from below, where LB(20) is SMA(20) minus 2 sigma
# 1. Sell when the closing price crosses UB(20) from above, where UB(20) is SMA(20) plus 2 sigma
# 1. No short-selling
# 
# The long-only BB(20, 2) is more difficult to implement than the long-only SMA(20) because we need to track buys and sells.
# For example, if the closing price is between LB(20) and BB(20), we need to know if our last trade was a buy or a sell.
# Further, if the closing price is below LB(20), we can still be long because we sell when the closing price crosses UB(20) from above.

# ### Implement a long-short RSI(14) strategy with Bitcoin

# From [Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/rsi):
# 
# > The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. Signals can be generated by looking for divergences and failure swings. RSI can also be used to identify the general trend.
# 
# Here is the RSI formula: $RSI(n) = 100 - \frac{100}{1 + RS(n)}$, where $RS(n) = \frac{SMA(U, n)}{SMA(D, n)}$.
# For "up days", $U = \Delta Adj\ Close$ and $D = 0$, and, for "down days", $U = 0$ and $D = - \Delta Adj\ Close$.
# Therefore, $U$ and $D$ are always non-negative.
# We can learn more about RSI [here](https://en.wikipedia.org/wiki/Relative_strength_index).
# 
# We will implement a long-short RSI(14) as follows:
# 
# 1. Enter a long position when  the RSI crosses 30 from below, and exit the position when the RSI crosses 50 from below
# 1. Enter a short position when the RSI crosses 70 from above, and exit the position when the RSI crosses 50 from above
