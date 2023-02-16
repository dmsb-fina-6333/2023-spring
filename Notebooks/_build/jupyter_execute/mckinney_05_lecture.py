#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 5 - Getting Started with pandas

# ## Introduction
# 
# Chapter 5 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) discusses the fundamentals of pandas, which will be our main tool for the rest of the semester.
# pandas is an abbrviation for *pan*el *da*ta, which provide time-stamped data for multiple individuals or firms.
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


# > pandas will be a major tool of interest throughout much of the rest of the book. It contains data structures and data manipulation tools designed to make data cleaning and analysis fast and easy in Python. pandas is often used in tandem with numerical computing tools like NumPy and SciPy, analytical libraries like statsmodels and scikit-learn, and data visualization libraries like matplotlib. pandas adopts significant parts of NumPy's idiomatic style of array-based computing, especially array-based functions and a preference for data processing without for loops. 
# >
# > While pandas adopts many coding idioms from NumPy, the biggest difference is that pandas is designed for working with tabular or heterogeneous data. NumPy, by contrast, is best suited for working with homogeneous numerical array data.
# 
# We will use pandas---a wrapper for NumPy that helps us manipulate and combine data---every day for the rest of the course.

# ## Introduction to pandas Data Structures
# 
# > To get started with pandas, you will need to get comfortable with its two workhorse data structures: Series and DataFrame. While they are not a universal solution for every problem, they provide a solid, easy-to-use basis for most applications.

# ### Series
# 
# > A Series is a one-dimensional array-like object containing a sequence of values (of similar types to NumPy types) and an associated array of data labels, called its index. The simplest Series is formed from only an array of data.
# 
# The early examples use integer and string labels, but date-time labels are most useful.

# In[3]:


obj = pd.Series([4, 7, -5, 3])
obj


# Contrast `obj` with a NumPy array equivalent:

# In[4]:


np.array([4, 7, -5, 3])


# In[5]:


obj.values


# In[6]:


obj.index  # similar to range(4)


# We did not explicitly assign an index to `obj`, so `obj` has an integer index that starts at 0.
# We can explicitly assign an index with the `index=` argument.

# In[7]:


obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[8]:


obj2.index


# In[9]:


obj2['a']


# In[10]:


obj2[2]


# In[11]:


obj2['d'] = 6
obj2


# In[12]:


obj2[['c', 'a', 'd']]


# A pandas series behaves like a NumPy array.
# We can use Boolean filters and perform vectorized mathematical operations.

# In[13]:


obj2 > 0


# In[14]:


obj2[obj2 > 0]


# In[15]:


obj2 * 2


# In[16]:


'b' in obj2


# In[17]:


'e' in obj2


# We can create a pandas series from a dictionary.
# The dictionary labels become the series index.

# In[18]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3


# We can create a pandas series from a list, too.
# Note that pandas respects the order of the assigned index.
# Also, pandas keeps California with `NaN` (not a number or missing value) and drops Utah because it was not in the index.

# In[19]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4


# Indices are one of pandas' super powers.
# When we perform mathematical operations, pandas aligns series by their indices.
# Here `NaN` is "not a number", which indicates missing values.
# `NaN` is considered a float, so the data type switches from int64 to float64.

# In[20]:


obj3 + obj4


# ### DataFrame
# 
# A pandas data frame is like a worksheet in an Excel workbook with row and columns that provide fast indexing.
# 
# > A DataFrame represents a rectangular table of data and contains an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc.). The DataFrame has both a row and column index; it can be thought of as a dict of Series all sharing the same index. Under the hood, the data is stored as one or more two-dimensional blocks rather than a list, dict, or some other collection of one-dimensional arrays. The exact details of DataFrame’s internals are outside the scope of this book.
# >
# > There are many ways to construct a DataFrame, though one of the most common is from a dict of equal-length lists or NumPy arrays:
# 

# In[21]:


data = {
    'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
    'year': [2000, 2001, 2002, 2001, 2002, 2003],
    'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
}
frame = pd.DataFrame(data)

frame


# We did not specify an index, so `frame` has the default index of integers starting at 0.

# In[22]:


frame2 = pd.DataFrame(
    data, 
    columns=['year', 'state', 'pop', 'debt'],
    index=['one', 'two', 'three', 'four', 'five', 'six']
)

frame2


# If we extract one column, via either `df.column` or `df['column']`, the result is a series.
# We can use either the `df.colname` or the `df['colname']` syntax to *extract* a column from a data frame as a series.
# ***However, we must use the `df['colname']` syntax to *add* a column to a data frame.***
# Also, we must use the `df['colname']` syntax to extract or add a column whose name contains a whitespace.

# In[23]:


frame2['state']


# In[24]:


frame2.state


# Similarly, if we extract one row. via either `df.loc['rowlabel']` or `df.iloc[rownumber]`, the result is a series.

# In[25]:


frame2.loc['one']


# Data frame have two dimensions, so we have to slice data frames more precisely than series.
# 
# 1. The `.loc[]` method slices by row labels and column names
# 1. The `.iloc[]` method slices by *integer* row and label indices

# In[26]:


frame2.loc['three']


# In[27]:


frame2.iloc[2]


# We can use NumPy's `[row, column]` syntanx with `.loc[]` and `.iloc[]`.

# In[28]:


frame2.loc['three', 'state'] # row, column


# In[29]:


frame2.loc['three', ['state', 'pop']] # row, column


# We can assign either scalars or arrays to data frame columns.
# 
# 1. Scalars will broadcast to every row in the data frame
# 1. Arrays must have the same length as the column

# In[30]:


frame2['debt'] = 16.5
frame2


# In[31]:


frame2['debt'] = np.arange(6.)
frame2


# If we assign a series to a data frame column, pandas will use the index to align it with the data frame.
# Data frame rows not in the series will be missing values `NaN`.

# In[32]:


val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
val


# In[33]:


frame2['debt'] = val
frame2


# We can add columns to our data frame, then delete them with `del`.

# In[34]:


frame2['eastern'] = (frame2.state == 'Ohio')
frame2


# In[35]:


del frame2['eastern']
frame2


# ### Index Objects

# In[36]:


obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index


# In[37]:


index[1:]


# Index objects are immutable!

# In[38]:


# index[1] = 'd'  # TypeError: Index does not support mutable operations


# Indices can contain duplicates, so an index does not guarantee our data are duplicate-free.

# In[39]:


dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])


# ## Essential Functionality
# 
# This section provides the most import pandas operations.
# It is difficult to provide an exhaustive reference, but this section provides a head start on the core pandas functionality.

# ### Dropping Entries from an Axis
# 
# > Dropping one or more entries from an axis is easy if you already have an index array or list without those entries. As that can require a bit of munging and set logic, the  drop method will return a new object with the indicated value or values deleted from an axis.

# In[40]:


obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[41]:


obj_without_d_and_c = obj.drop(['d', 'c'])
obj_without_d_and_c


# The `.drop()` method works on data frames, too.

# In[42]:


data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)

data


# In[43]:


data.drop(['Colorado', 'Ohio']) # implied ", axis=0"


# In[44]:


data.drop(['Colorado', 'Ohio'], axis=0)


# In[45]:


data.drop(index=['Colorado', 'Ohio'])


# The `.drop()` method accepts an `axis` argument and the default is `axis=0` to drop rows based on labels.
# To drop columns, we use `axis=1` or `axis='columns'`.

# In[46]:


data.drop('two', axis=1)


# In[47]:


data.drop(columns='two')


# ### Indexing, Selection, and Filtering
# 
# Indexing, selecting, and filtering will be among our most-used pandas features.
# 
# > Series indexing (obj[...]) works analogously to NumPy array indexing, except you can use the Series's index values instead of only integers.  

# In[48]:


obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj


# In[49]:


obj['b']


# In[50]:


obj[1]


# The code directly above works.
# However, I prefer to be explicit and use `.iloc[]` when I index or slice by integers.

# In[51]:


obj.iloc[1]


# In[52]:


obj.iloc[1:3]


# ***When we slice with labels, the left and right endpoints are inclusive.***

# In[53]:


obj['b':'c']


# In[54]:


obj['b':'c'] = 5
obj


# In[55]:


data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'New York'],
    columns=['one', 'two', 'three', 'four']
)

data


# Indexing one column returns a series.

# In[56]:


data['two']


# Indexing two or more columns returns a data frame.

# In[57]:


data[['three', 'one']]


# If we want a one-column data frame, we can use `[[]]`:

# In[58]:


data['three']


# In[59]:


data[['three']]


# When we slice with integer indices with `[]`, we slice rows.

# In[60]:


data[:2]


# When I slice rows, I prefer to use `.loc[]` or `.iloc[]` to avoid confusion.

# In[61]:


data.iloc[:2]


# We can index a data frame with Booleans, as we did with NumPy arrays.

# In[62]:


data < 5


# In[63]:


data[data < 5] = 0
data


# Finally, we can chain slices.

# In[64]:


data.iloc[:, :3][data.three > 5]


# ***Table 5-4*** summarizes data frame indexing and slicing options:
# 
# - `df[val]`: Select single column or sequence of columns from the DataFrame; special case conveniences: boolean array (filter rows), slice (slice rows), or boolean DataFrame (set values based on some criterion)
# - `df.loc[val]`: Selects single row or subset of rows from the DataFrame by label
# - `df.loc[:, val]`: Selects single column or subset of columns by label
# - `df.loc[val1, val2]`: Select both rows and columns by label
# - `df.iloc[where]`: Selects single row or subset of rows from the DataFrame by integer position
# - `df.iloc[:, where]`: Selects single column or subset of columns by integer position
# - `df.iloc[where_i, where_j]`: Select both rows and columns by integer position
# - `df.at[label_i, label_j]`: Select a single scalar value by row and column label
# - `df.iat[i, j]`: Select a single scalar value by row and column position (integers) reindex method Select either rows or columns by labels
# - `get_value`, `set_value` methods: Select single value by row and column label
# 
# pandas is powerful and these options can be overwhelming!
# We will typically use `df[val]` to select columns (here `val` is either a string or list of strings), `df.loc[val]` to select rows (here `val` is a row label), and `df.loc[val1, val2]` to select both rows and columns.
# The other options add flexibility, and we may occasionally use them.
# However, our data will be large enough that counting row and column number will be tedious, making `.iloc[]` impractical.

# ### Integer Indexes

# In[65]:


ser = pd.Series(np.arange(3.))
ser


# The following indexing yields an error because the series cannot fall back to NumPy array indexing.
# Falling back to NumPy array indexing here would generate many subtle bugs elsewhere.

# In[66]:


# ser[-1]


# In[67]:


ser.iloc[-1]


# However, the following indexing works fine because with string labels there is no ambiguity.

# In[68]:


ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2


# In[69]:


ser2[-1]


# In[70]:


ser2.iloc[-1]


# In practice, these errors will not be an issue because we will index or slice with stock identifiers and dates instead of integers.
# To avoid condusion, we should use `.iloc[]` to index or slice with integers.

# ### Arithmetic and Data Alignment
# 
# > An important pandas feature for some applications is the behavior of arithmetic between objects with different indexes. When you are adding together objects, if any index pairs are not the same, the respective index in the result will be the union of the index pairs. For users with database experience, this is similar to an automatic outer join on the index labels. 

# In[71]:


s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])


# In[72]:


s1


# In[73]:


s2


# In[74]:


s1 + s2


# In[75]:


df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])


# In[76]:


df1


# In[77]:


df2


# In[78]:


df1 + df2


# In[79]:


df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})


# In[80]:


df1


# In[81]:


df2


# In[82]:


df1 - df2


# #### Arithmetic methods with fill values

# In[83]:


df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df2.loc[1, 'b'] = np.nan


# In[84]:


df1


# In[85]:


df2


# In[86]:


df1 + df2


# We can specify a fill value for `NaN` values.
# Note that pandas fills would-be `NaN` values in each data frame *before* the arithmetic operation.

# In[87]:


df1.add(df2, fill_value=0)


# #### Operations between DataFrame and Series

# In[88]:


arr = np.arange(12.).reshape((3, 4))
arr


# In[89]:


arr[0]


# In[90]:


arr - arr[0]


# Arithmetic operations between series and data frames behave the same as the example above.

# In[91]:


frame = pd.DataFrame(
    np.arange(12.).reshape((4, 3)),
    columns=list('bde'),
    index=['Utah', 'Ohio', 'Texas', 'Oregon']
)

series = frame.iloc[0]


# In[92]:


frame


# In[93]:


series


# In[94]:


frame - series


# In[95]:


series2 = pd.Series(range(3), index=['b', 'e', 'f'])


# In[96]:


frame


# In[97]:


series2


# In[98]:


frame + series2


# In[99]:


series3 = frame['d']


# In[100]:


frame.sub(series3, axis='index')


# ### Function Application and Mapping

# In[101]:


np.random.seed(42)
frame = pd.DataFrame(
    np.random.randn(4, 3), 
    columns=list('bde'),
    index=['Utah', 'Ohio', 'Texas', 'Oregon']
)

frame


# In[102]:


frame.abs()


# > Another frequent operation is applying a function on one-dimensional arrays to each column or row. DataFrame’s apply method does exactly this:

# Note that we can use anonymous (lambda) functions "on the fly":

# In[103]:


frame.apply(lambda x: x.max() - x.min())


# In[104]:


frame.apply(lambda x: x.max() - x.min(), axis=1)


# However, under the hood, the `.apply()` is basically a `for` loop and much slowly than optimized, built-in methods.
# Here is an example of the speed costs of `.apply()`:

# In[105]:


get_ipython().run_line_magic('timeit', "frame['e'].abs()")


# In[106]:


get_ipython().run_line_magic('timeit', "frame['e'].apply(np.abs)")


# ## Summarizing and Computing Descriptive Statistics

# In[107]:


df = pd.DataFrame(
    [[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
    index=['a', 'b', 'c', 'd'],
    columns=['one', 'two']
)

df


# In[108]:


df.sum()


# In[109]:


df.sum(axis=1)


# In[110]:


df.mean(axis=1, skipna=False)


# The `.idxmax()` method returns the label for the maximum observation.

# In[111]:


df.idxmax()


# The `.describe()` returns summary statistics for each numerical column in a data frame.

# In[112]:


df.describe()


# For non-numerical data, `.describe()` returns alternative summary statistics.

# In[113]:


obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()


# ### Correlation and Covariance

# To explore correlation and covariance methods, we can use Yahoo! Finance stock data.
# We can use the yfinance package to import these data.
# We can use the requests-cache package to cache our data requests, which avoid unnecessarily re-downloading data.
# 
# We can install these two functions with the `%pip` magic:

# In[114]:


# %pip install yfinance requests-cache


# If we are running Python locally, we only need to run the `%pip` magic once.
# If we are running Python in the cloud, we may need to run the `%pip` magic once *per login*.

# In[115]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# In[116]:


tickers = yf.Tickers('AAPL IBM MSFT GOOG', session=session)


# In[117]:


prices = tickers.history(period='max', auto_adjust=False, progress=False)


# In[118]:


prices.index = prices.index.tz_localize(None)


# In[119]:


prices['Adj Close']


# The `prices` data frames contains daily data for AAPL, IBM, MSFT, and GOOG.
# The `Adj Close` column provides a reverse-engineered daily closing price that accounts for dividends paid and stock splits (and reverse splits).
# As a result, the `.pct_change()` in `Adj Close` considers both price changes (i.e., capital gains) and dividends, so $R_t = \frac{(P_t + D_t) - P_{t-1}}{P_{t-1}} = \frac{\text{Adj Close}_t - \text{Adj Close}_{t-1}}{\text{Adj Close}_{t-1}}.$

# In[120]:


returns = prices['Adj Close'].pct_change().dropna()
returns


# We multiply by 252 to annualize mean daily returns because means grow linearly with time and there are (about) 252 trading days per year.

# In[121]:


returns.mean().mul(252)


# We multiply by $\sqrt{252}$ to annualize the standard deviation of daily returns because variances grow linearly with time, there are (about) 252 trading days per year, and the standard deviation is the square root of the variance.

# In[122]:


returns.std().mul(np.sqrt(252))


# ***The best explanation I have found on why stock return volatility (the standard deviation of stocks returns) grows with the square root of time is at the bottom of page 7 of [chapter 8 of Ivo Welch's free corporate finance textbook](https://book.ivo-welch.info/read/source5.mba/08-invchoice.pdf).***

# We can calculate pairwise correlations.

# In[123]:


returns['MSFT'].corr(returns['IBM'])


# We can also calculate correlation matrices.

# In[124]:


returns.corr()


# In[125]:


returns.corr().loc['MSFT', 'IBM']


# In[ ]:




