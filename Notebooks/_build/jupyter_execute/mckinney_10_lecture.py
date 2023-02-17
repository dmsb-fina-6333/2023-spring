#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Data Aggregation and Group Operations

# ## Introduction

# Chapter 10 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) discusses groupby operations, which are the pandas equivalent of Excel pivot tables.
# Pivot tables help us calculate statistics (e.g., sum, mean, and median) for one set of variables by groups of other variables (e.g., weekday or ticker).
# For example, we could use a pivot table to calculate mean daily stock returns by weekday.
# 
# We will focus on:
# 
# 1. Using `.groupby()` to group by columns, indexes, and functions
# 1. Using `.agg()` to aggregate multiple functions
# 1. Using pivot tables as an alternative to `.groupby()`
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


# ## GroupBy Mechanics

# "Split-apply-combine" is an excellent way to describe and visualize pandas groupby operations.
# 
# > Hadley Wickham, an author of many popular packages for the R programming 
# language, coined the term split-apply-combine for describing group operations. In the
# first stage of the process, data contained in a pandas object, whether a Series, DataFrame, or otherwise, is split into groups based on one or more keys that you provide.
# The splitting is performed on a particular axis of an object. For example, a DataFrame
# can be grouped on its rows (axis=0) or its columns (axis=1). Once this is done, a
# function is applied to each group, producing a new value. Finally, the results of all
# those function applications are combined into a result object. The form of the resulting object will usually depend on what’s being done to the data. See Figure 10-1 for a
# mockup of a simple group aggregation.
# 
# Figure 10-1 visualizes a split-apply-combine operation that:
# 
# 1. Splits by the `key` column (i.e., "groups by `key`")
# 2. Applies the sum operation to the `data` column (i.e., "and sums `data`")
# 3. Combines the grouped sums
# 
# I describe this operation as "sum the `data` column by groups formed on the `key` column."

# In[4]:


np.random.seed(42)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})

df


# Here is one way to calculate the means of `data1` by groups formed on `key1`.

# In[5]:


df.loc[df['key1'] == 'a', 'data1'].mean()


# In[6]:


df.loc[df['key1'] == 'b', 'data1'].mean()


# We can do this calculation more quickly!
# 
# 1. Use the `.groupby()` method to group by `key1`
# 2. Use the `.mean()` method to sum `data1` within each value of `key1`
# 
# Note that without the `.mean()` method, pandas only sets up the grouped object, which can accept the `.mean()` method.

# In[7]:


grouped = df['data1'].groupby(df['key1'])
grouped


# In[8]:


grouped.mean()


# We can can chain the `.groupby()` and `.mean()` methods!

# In[9]:


df['data1'].groupby(df['key1']).mean()


# If we prefer our result as a dataframe instead of a series, we can wrap `data1` with two sets of square brackets.

# In[10]:


df[['data1']].groupby(df['key1']).mean()


# We can group by more than one variable.
# We get a hierarchical row index (or row multi-index) when we group by more than one variable.

# In[11]:


means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means


# We can use the `.unstack()` method if we want to use both rows and columns to organize data.
# Recall the `.unstack()` method un-stacks the inner index level (i.e., `level = -1`) by default so that `key2` values become the columns.

# In[12]:


means.unstack()


# The grouping variables can be columns in the data frame we want to group with the `.groupby()` method.
# Our grouping variables are typically columns in the data frame we want to group, so this syntax is more compact and easier to understand.

# In[13]:


df.groupby('key1').mean()


# In[14]:


df.groupby(['key1', 'key2']).mean()


# We can use tab completion to reminder ourselves of methods we can apply to grouped series and data frames.

# ### Iterating Over Groups

# We can iterate over groups, too, because the `.groupby()` method generates a sequence of tuples.
# Each tuples contains the value(s) of the grouping variable(s) and its chunk of the dataframe.
# McKinney provides two loops to show how to iterate over groups.

# In[15]:


for name, group in df.groupby('key1'):
    print(name, group, sep='\n')


# In[16]:


for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2), group, sep='\n')


# ### Selecting a Column or Subset of Columns

# We preview the idea of grouping an entire dataframe above.
# However, I want to explain McKinney's use of the phrase "syntactic sugar."
# Here is the context:
# 
# > Indexing a GroupBy object created from a DataFrame with a column name or array
# of column names has the effect of column subsetting for aggregation. This means
# that:
# >
# > ```python
# > df.groupby('key1')['data1']
# > df.groupby('key1')[['data2']]
# > ```
# >
# > are syntactic sugar for
# >
# > ```python
# > df['data1'].groupby(df['key1'])
# > df[['data2']].groupby(df['key1'])
# > ```
# 
# "Syntactic sugar" makes code easier to type or read without adding functionality.
# It makes code "sweeter" for humans to type or read by making it more concise or clear.
# The implication is that syntactic sugar makes code faster to type/read but does make code faster to execute.

# In[17]:


df.groupby(['key1', 'key2'])[['data2']].mean()


# ### Grouping with Functions

# We can also group with functions.
# Below, we group with the `len` function, which calculates the length of the first names in the row index.
# We could instead add a helper column to `people`, but it is easier to pass a function to `.groupby()`.

# In[18]:


np.random.seed(42)
people = pd.DataFrame(
    data=np.random.randn(5, 5), 
    columns=['a', 'b', 'c', 'd', 'e'], 
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
)

people


# In[19]:


people.groupby(len).sum()


# We can mix functions, lists, dictionaries, etc. that we pass to `.groupby()`.

# In[20]:


key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# In[21]:


d = {'Joe': 'a', 'Jim': 'b'}
people.groupby([len, d]).min()


# In[22]:


d_2 = {'Joe': 'Cool', 'Jim': 'Nerd', 'Travis': 'Cool'}
people.groupby([len, d_2]).min()


# ### Grouping by Index Levels

# We can also group by index levels.
# We can specify index levels by either level number or name.

# In[23]:


columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]],
                                    names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)

hier_df


# In[24]:


hier_df.groupby(level='cty', axis=1).count()


# In[25]:


hier_df.groupby(level='cty', axis='columns').count()


# In[26]:


hier_df.groupby(level='tenor', axis=1).count()


# ## Data Aggregation

# Table 10-1 provides the optimized groupby methods:
# 
# - `count`: Number of non-NA values in the group
# - `sum`: Sum of non-NA values
# - `mean`: Mean of non-NA values
# - `median`: Arithmetic median of non-NA values
# - `std`, `var`: Unbiased (n – 1 denominator) standard deviation and variance
# - `min`, `max`: Minimum and maximum of non-NA values
# - `prod`: Product of non-NA values
# - `first`, `last`: First and last non-NA values
# 
# These optimized methods are fast and efficient, but pandas lets us use other, non-optimized methods.
# First, any series method is available.

# In[27]:


df.groupby('key1')['data1'].quantile(0.9)


# Second, we can write our own functions and pass them to the `.agg()` method.
# These functions should accept an array and returns a single value.

# In[28]:


def max_minus_min(arr):
    return arr.max() - arr.min()


# In[29]:


df.sort_values(by=['key1', 'data1'])


# In[30]:


df.groupby('key1')['data1'].agg(max_minus_min)


# Some other methods work, too, even if they are do not aggregate an array to a single value.

# In[31]:


df.groupby('key1')['data1'].describe()


# ### Column-Wise and Multiple Function Application

# The `.agg()` methods provides two more handy features:
# 
# 1. We can pass multiple functions to operate on all of the columns
# 2. We can pass specific functions to operate on specific columns

# Here is an example with multiple functions:

# In[32]:


df.groupby('key1')['data1'].agg(['mean', 'median', 'min', 'max'])


# In[33]:


df.groupby('key1')[['data1', 'data2']].agg(['mean', 'median', 'min', 'max'])


# What if I wanted to calculate the mean of `data1` and the median of `data2` by `key1`?

# In[34]:


df.groupby('key1').agg({'data1': 'mean', 'data2': 'median'})


# What if I wanted to calculate the mean *and standard deviation* of `data1` and the median of `data2` by `key1`?

# In[35]:


df.groupby('key1').agg({'data1': ['mean', 'std'], 'data2': 'median'})


# ## Apply: General split-apply-combine

# The `.agg()` method aggrates an array to a single value.
# We can use the `.apply()` method for more general calculations.
# 
# We can combine the `.groupby()` and `.apply()` methods to:
# 
# 1. Split a dataframe by grouping variables
# 2. Call the applied function on each chunk of the original dataframe
# 3. Recombine the output of the applied function

# In[36]:


def top(x, col, n=1):
    return x.sort_values(col).head(n)


# In[37]:


df.groupby('key1').apply(top, col='data1')


# In[38]:


df.groupby('key1').apply(top, col='data1', n=2)


# ## Pivot Tables and Cross-Tabulation

# Above we manually made pivot tables with the `groupby()`, `.agg()`, `.apply()` and `.unstack()` methods.
# pandas provides a literal interpreation of Excel-style pivot tables with the `.pivot_table()` method and the `pandas.pivot_table()` function.
# These also provide row and column totals via "margins".
# It is worthwhile to read-through the `.pivot_table()` docstring several times.

# In[39]:


ind = (
    yf.download(
        tickers='^GSPC ^DJI ^IXIC ^FTSE ^N225 ^HSI',
        progress=False
    )
    .rename_axis(columns=['Variable', 'Index'])
    .stack()
)

ind.head()


# The default aggregation function for `.pivot_table()` is `mean`.

# In[40]:


ind.loc['2015':].pivot_table(index='Index')


# In[41]:


ind.loc['2015':].pivot_table(index='Index', aggfunc='median')


# We can use 
#     `values` to select specific variables, 
#     `pd.Grouper()` to sample different date windows, 
#     and 
#     `aggfunc` to select specific aggregation functions.

# In[42]:


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


# In[ ]:




