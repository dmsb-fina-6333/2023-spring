#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 8 - Data Wrangling: Join, Combine, and Reshape

# ## Introduction

# Chapter 8 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) introduces a few important pandas concepts:
# 
# 1. Joining or merging is combining 2+ data frames on 1+ indexes or columns into 1 data frame
# 1. Reshaping is rearranging data frames so it has fewer columns and more rows (wide to long) or more columns and fewer rows (long to wide); we can also reshape a series to a data frame and vice versa
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


# ## Hierarchical Indexing

# We need to learn about hierarchical indexing before we learn about combining and reshaping data.
# A hierarchical index gives two or more index levels to an axis.
# For example, we could index rows by ticker and date.
# Or we could index columns by variable and ticker.
# Hierarchical indexing helps us work with high-dimensional data in a low-dimensional form.

# In[4]:


np.random.seed(42)
data = pd.Series(
    data=np.random.randn(9),
    index=[
        ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        [1, 2, 3, 1, 3, 1, 2, 2, 3]
    ]
)

data


# We can partially index this series to concisely subset data.

# In[5]:


data['b']


# In[6]:


data['b':'c']


# In[7]:


data.loc[['b', 'd']]


# We can subset on the index inner level, too.
# Here the first `:` slices all values in the outer index.

# In[8]:


data.loc[:, 2]


# Here `data` has a stacked format.
# For each outer index level (the letters), we have multiple observations based on the inner index level (the numbers).
# We can un-stack `data` to convert the inner index level to columns.

# In[9]:


data.unstack()


# In[10]:


data.unstack().stack()


# We can create a data frame with hieracrhical indexes or multi-indexes on rows *and* columns.

# In[11]:


frame = pd.DataFrame(
    data=np.arange(12).reshape((4, 3)),
    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
    columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']]
)
frame


# We can name these multi-indexes but names are not required.

# In[12]:


frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame


# Recall that `df[val]` selects the `val` column.
# Here `frame` has a multi-index for the columns, so `frame['Ohio']` selects all columns with Ohio as the outer index level.

# In[13]:


frame['Ohio']


# We can pass a tuple if we only want one column. 

# In[14]:


frame[[('Ohio', 'Green')]]


# We have to do a more work to slice the inner level of the column index.

# In[15]:


frame.loc[:, (slice(None), 'Green')]


# We can use `pd.IndexSlice[:, 'Green']` an alternative to `(slice(None), 'Green')`.

# In[16]:


frame.loc[:, pd.IndexSlice[:, 'Green']]


# ### Reordering and Sorting Levels

# We can swap index levels with the `.swaplevel()` method.
# The default arguments are `i=-2` and `j=-1`, which swap the two innermost index levels.

# In[17]:


frame.swaplevel()


# We can use index *names*, too.

# In[18]:


frame.swaplevel('key1', 'key2')


# We can also sort on an index (or list of indexes).
# After we swap levels, we may want to sort our data.

# In[19]:


frame


# In[20]:


frame.sort_index(level=1)


# Again, we can give index *names*, too.

# In[21]:


frame.sort_index(level='key2')


# We can sort by two or more index levels by passing a list of index levels or names.

# In[22]:


frame.sort_index(level=[0, 1])


# We can chain these methods, too.

# In[23]:


frame.swaplevel(0, 1).sort_index(level=0)


# ### Indexing with a DataFrame's columns

# We can convert a column into an index and an index into a column with the `.set_index()` and `.reset_index()` methods.

# In[24]:


frame = pd.DataFrame({
    'a': range(7), 
    'b': range(7, 0, -1),
    'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
    'd': [0, 1, 2, 0, 1, 2, 3]
})
frame


# The `.set_index()` method converts columns to indexes, and removes the columns from the data frame by default.

# In[25]:


frame2 = frame.set_index(['c', 'd'])
frame2


# The `.reset_index()` method removes the indexes, adds them as columns, and sets in integer index.

# In[26]:


frame2.reset_index()


# ## Combining and Merging Datasets

# pandas provides several methods and functions to combine and merge data.
# We can typically create the same output with any of these methods or functions, but one may be more efficient than the others.
# If I want to combine data frames with similar indexes, I try the `.join()` method first.
# The `.join()` also lets use can combine more than two data frames at once.
# Otherwise, I try the `.merge()` method, which has a function `pd.merge()`, too.
# The `pd.merge()` function is more general than the `.join()` method, so we will start with `pd.merge()`.
# 
# The [pandas website](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#) provides helpful visualizations.

# ### Database-Style DataFrame Joins

# > Merge or join operations combine datasets by linking rows using one or more keys. These operations are central to relational databases (e.g., SQL-based). The merge function in pandas is the main entry point for using these algorithms on your data.
# 
# We will start with the `pd.merge()` syntax, but pandas also has `.merge()` and `.join()` methods.
# Learning these other syntaxes is easy once we understand the `pd.merge()` syntax.

# In[27]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})


# In[28]:


df1


# In[29]:


df2


# In[30]:


pd.merge(df1, df2)


# The default `how` is `how='inner'`, so `pd.merge()` inner joins left and right data frames by default, keeping only rows that appear in both.
# We can specify `how='outer'`, so `pd.merge()` outer joins left and right data frames, keeping all rows that appear in either.

# In[31]:


pd.merge(df1, df2, how='outer')


# A left merge keeps only rows that appear in the left data frame.

# In[32]:


pd.merge(df1, df2, how='left')


# A rights merge keeps only rows that appear in the right data frame.

# In[33]:


pd.merge(df1, df2, how='right')


# By default, `pd.merge()` merges on all columns that appear in both data frames.
# 
# > `on` : label or list
#     Column or index level names to join on. These must be found in both
#     DataFrames. If `on` is None and not merging on indexes then this defaults
#     to the intersection of the columns in both DataFrames.
#     
# Here `key` is the only common column between `df1` and `df2`.
# We *should* specify `on` to avoid unexpected results.

# In[34]:


pd.merge(df1, df2, on='key')


# We *must* specify `left_on` and `right_on` if our left and right data frames do not have a common column.

# In[35]:


df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})


# In[36]:


df3


# In[37]:


df4


# In[38]:


# pd.merge(df3, df4) # this code fails/errors because there are not common columns


# In[39]:


pd.merge(df3, df4, left_on='lkey', right_on='rkey')


# Here `pd.merge()` dropped row `c` from `df3` and row `d` from `df4`.
# Rows `c` and `d` dropped because `pd.merge()` *inner* joins be default.
# An inner join keeps the intersection of the left and right data frame keys.
# Further, rows `a` and `b` from `df4` appear three times to match `df3`.
# If we want to keep rows `c` and `d`, we can *outer* join `df3` and `df4` with `how='outer'`.

# In[40]:


pd.merge(df1, df2, how='outer')


# > Many-to-many merges have well-defined, though not necessarily intuitive, behavior.

# In[41]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})


# In[42]:


df1


# In[43]:


df2


# In[44]:


pd.merge(df1, df2, on='key')


# > Many-to-many joins form the Cartesian product of the rows. Since there were three `b` rows in the left DataFrame and two in the right one, there are six `b` rows in the result. The join method only affects the distinct key values appearing in the result.
# 
# Be careful with many-to-many joins!
# In finance, we do not expect many-to-many joins because we expect at least one of the data frames to have unique observations.
# ***pandas will not warn us if we accidentally perform a many-to-many join instead of a one-to-one or many-to-one join.***

# We can merge on more than one key.
# For example, we may merge two data sets on ticker-date pairs or industry-date pairs.

# In[45]:


left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                     'key2': ['one', 'two', 'one'],
                     'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                      'key2': ['one', 'one', 'one', 'two'],
                      'rval': [4, 5, 6, 7]})


# In[46]:


left


# In[47]:


right


# In[48]:


pd.merge(left, right, on=['key1', 'key2'], how='outer')


# When column names overlap between the left and right data frames, `pd.merge()` appends `_x` and `_y` to the left and right versions of the overlapping column names.

# In[49]:


pd.merge(left, right, on='key1')


# I typically specify suffixes to avoid later confusion.

# In[50]:


pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


# I read the `pd.merge()` docstring whenever I am in doubt.
# ***Table 8-2*** lists the most commonly used arguments for `pd.merge()`.
# 
# > - `left`: DataFrame to be merged on the left side.
# > - `right`: DataFrame to be merged on the right side.
# > - `how`: One of 'inner', 'outer', 'left', or 'right'; defaults to 'inner'.
# > - `on`: Column names to join on. Must be found in both DataFrame objects. If not specified and no other join keys given will use the intersection of the column names in left and right as the join keys.
# > - `left_on`: Columns in left DataFrame to use as join keys.
# > - `right_on`: Analogous to left_on for left DataFrame.
# > - `left_index`: Use row index in left as its join key (or keys, if a MultiIndex).
# > - `right_index`: Analogous to left_index.
# > - `sort`: Sort merged data lexicographically by join keys; True by default (disable to get better performance in some cases on large datasets).
# > - `suffixes`: Tuple of string values to append to column names in case of overlap; defaults to ('_x', '_y') (e.g., if 'data' in both DataFrame objects, would appear as 'data_x' and 'data_y' in result).
# > - `copy`: If False, avoid copying data into resulting data structure in some exceptional cases; by default always copies.
# > - `indicator`: Adds a special column _merge that indicates the source of each row; values will be 'left_only', 'right_only', or 'both' based on the origin of the joined data in each row.

# ### Merging on Index

# If we want to use `pd.merge()` to join on row indexes, we can use the `left_index` and `right_index` arguments.

# In[51]:


left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])


# In[52]:


left1


# In[53]:


right1


# In[54]:


pd.merge(left1, right1, left_on='key', right_index=True, how='outer')


# The index arguments work for hierarchical indexes (multi indexes), too.

# In[55]:


lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                      'key2': [2000, 2001, 2002, 2001, 2002],
                      'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                      index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                             [2001, 2000, 2000, 2000, 2001, 2002]],
                      columns=['event1', 'event2'])


# In[56]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')


# In[57]:


left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])


# If we use both left and right indexes, `pd.merge()` will keep the index.

# In[58]:


pd.merge(left2, right2, how='outer', left_index=True, right_index=True)


# > DataFrame has a convenient join instance for merging by index. It can also be used to combine together many DataFrame objects having the same or similar indexes but non-overlapping columns.
# 
# If we have matching indexes on left and right, we can use `.join()`.

# In[59]:


left2


# In[60]:


right2


# In[61]:


left2.join(right2, how='outer')


# The `.join()` method left joins by default.
# The `.join()` method uses indexes, so it requires few arguments and accepts a list of data frames.

# In[62]:


another = pd.DataFrame(
    data=[[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
    index=['a', 'c', 'e', 'f'],
    columns=['New York', 'Oregon']
)

another


# In[63]:


left2.join([right2, another])


# In[64]:


left2.join([right2, another], how='outer')


# ### Concatenating Along an Axis

# The `pd.concat()` function provides a flexible way to combine data frames and series along either axis.
# I typically use `pd.concat()` to combine:
# 
# 1. A list of data frames with similar layouts
# 1. A list of series because series do not have `.join()` or `.merge()` methods
# 
# The first is handy if we have to read and combine a directory of .csv files.

# In[65]:


s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])


# In[66]:


pd.concat([s1, s2, s3])


# In[67]:


pd.concat([s1, s2, s3], axis=1)


# In[68]:


result = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])

result


# In[69]:


result.unstack()


# In[70]:


pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])


# In[71]:


df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'], columns=['three', 'four'])


# In[72]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])


# In[73]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])


# ## Reshaping and Pivoting

# Above, we briefly explore reshaping data with `.stack()` and `.unstack()`.
# Here we explore reshaping data more deeply.

# ### Reshaping with Hierarchical Indexing

# Hierarchical indexes (multi-indexes) help reshape data.
# 
# > There are two primary actions:
# > - stack: This "rotates" or pivots from the columns in the data to the rows
# > - unstack: This pivots from the rows into the columns

# In[74]:


data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))

data


# In[75]:


result = data.stack()

result


# In[76]:


result.unstack()


# In[77]:


s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])

data2


# In[78]:


data2.unstack()


# Un-stacking may introduce missing values because data frames are rectangular.

# By default, stacking drops these missing values.

# In[79]:


data2.unstack().stack()


# However, we can keep missing values with `dropna=False`.

# In[80]:


data2.unstack().stack(dropna=False)


# In[81]:


df = pd.DataFrame({
    'left': result, 
    'right': result + 5
    },
    columns=pd.Index(['left', 'right'], name='side')
)

df


# If we un-stack a data frame, the un-stacked level becomes the innermost level in the resulting index.

# In[82]:


df.unstack('state')


# We can chain `.stack()` and `.unstack()` to rearrange our data.

# In[83]:


df.unstack('state').stack('side')


# McKinney provides two more subsections on reshaping data with the `.pivot()` and `.melt()` methods.
# Unlike, the stacking methods, the pivoting methods can aggregate data and do not require an index.
# We will skip these additional aggregation methods for now.
