#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 4 - NumPy Basics: Arrays and Vectorized Computation

# ## Introduction

# Chapter 4 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) discusses the NumPy package (an abbreviation of numerical Python), which is the foundation for numerical computing in Python, including pandas.
# 
# We will focus on:
# 
# 1. Creating arrays
# 1. Slicing arrays
# 1. Applying functions and methods to arrays
# 1. Using conditional logic with arrays (i.e., `np.where()` and `np.select()`)
# 
# ***Note:*** 
# Indented block quotes are from McKinney unless otherwise indicated.
# The section numbers here differ from McKinney because we will only discuss some topics.

# The typical abbreviation for NumPy is `np`.

# In[1]:


import numpy as np


# The "magic" function `%precision 4` tells JupyterLab to print NumPy arrays to 4 decimals.
# This magic function only changes the printed precision and does not change the stored precision of the underlying values.

# In[2]:


get_ipython().run_line_magic('precision', '4')


# McKinney thoroughly discuesses the history on NumPy, as well as its technical advantages.
# But here is a simple illustration of the speed and syntax advantages of NumPy of Python's built-in data structures.
# First, we create a list and array with values from 0 to 999,999.

# In[3]:


my_list = list(range(1_000_000))


# In[4]:


my_arr = np.arange(1_000_000)


# In[5]:


my_list[:5]


# In[6]:


my_arr[:5]


# If we want to double each of the values in `my_list` we have to use a for loop or a list comprehension.

# In[7]:


# my_list * 2 # concatenates two copies of my_list


# In[8]:


# [2 * x for x in my_list] # list comprehension to double each value


# However, we can just multiply `my_arr` by two because math "just works" with NumPy.

# In[9]:


my_arr * 2


# We can use the "magic" function `%timeit` to time these two calculations.

# In[10]:


get_ipython().run_line_magic('timeit', '[x * 2 for x in my_list]')


# In[11]:


get_ipython().run_line_magic('timeit', 'my_arr * 2')


# The NumPy version is few hundred times faster than the list version.
# The NumPy version is also faster to type, read, and troubleshoot, which are typically more important.
# Our time is more valuable than the computer time!

# ## The NumPy ndarray: A Multidimensional Array Object

# > One of the key features of NumPy is its N-dimensional array object, or ndarray, which is a fast, flexible container for large datasets in Python. Arrays enable you to perform mathematical operations on whole blocks of data using similar syntax to the equivalent operations between scalar elements.
# 
# We can generate random data to explore NumPy arrays.
# Whenever we use random data, we should set the random number seed with `np.random.seed(42)`, which makes our random numbers repeatable.
# Because we set the random number seed just before we generate `data`, your `data` and my `data` will be identical.

# In[12]:


np.random.seed(42)
data = np.random.randn(2, 3)


# In[13]:


data


# Multiplying `data` by 10 multiplies each element in `data` by 10, and adding `data` to itself does element-wise addition.
# The compromise to achieve this common-sense behavior is that NumPy arrays must contain homogeneous data types (e.g., all floats or all integers).

# In[14]:


data * 10


# Addition in NumPy is also elementwise.

# In[15]:


data_2 = data + data


# NumPy arrays have attributes.
# Recall that Jupyter Notebooks provides tab completion.

# In[16]:


data.ndim


# In[17]:


data.shape


# In[18]:


data.dtype


# We access or slice elements in a NumPy array the same way as lists and tuples: `[]`

# In[19]:


data[0]


# We can chain `[]`s.

# In[20]:


data[0][0] # zero row, then the zero element in the zero row


# However, with NumPy arrays. we can combine a chain inside one pair of `[]`s.
# For example, `[i][j]` becomes `[i, j]`, `[i][j][k]` becomes `[i, j, k]`.
# This abbreviated notation is similar to what you will see in math and econometrics courses.

# In[21]:


data[0, 0] # zero row, zero column


# ### Creating ndarrays

# > The easiest way to create an array is to use the array function. This accepts any sequence-like object (including other arrays) and produces a new NumPy array containing the passed data

# In[22]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)


# In[23]:


arr1


# In[24]:


arr1.dtype


# Here `np.array()` casts the values in `data1` to floats becuase NumPy arrays must have homogenous data types.
# We could coerce these values to integers but would lose information.

# In[25]:


np.array(data1, dtype=np.int64)


# We can coerce or cast a list-of-lists to a two-dimensional NumPy array.

# In[26]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)


# In[27]:


arr2


# In[28]:


arr2.ndim


# In[29]:


arr2.shape


# In[30]:


arr2.dtype


# There are several other ways to create NumPy arrays.

# In[31]:


np.zeros(10)


# In[32]:


np.zeros((3, 6))


# The `np.arange()` function is similar to Python's built-in `range()`, but it creates an array directly.

# In[33]:


np.array(range(15))


# In[34]:


np.arange(15)


# ***Table 4-1*** from McKinney lists some NumPy array creation functions.
# 
# - `array`: Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a dtype or explicitly specifying a dtype; copies the input data by default
# - `asarray`:  Convert input to ndarray, but do not copy if the input is already an ndarray 
# - `arange`:  Like the built-in range but returns an ndarray instead of a list
# - `ones`, `ones_like`:  Produce an array of all 1s with the given shape and dtype; `ones_like` takes another array and produces a `ones` array of the - same shape and dtype
# - `zeros`, `zeros_like`:  Like `ones` and `ones_like` but producing arrays of 0s instead
# - `empty`, `empty_like`:  Create new arrays by allocating new memory, but do not populate with any values like ones and zeros
# - `full`, `full_like`:  Produce an array of the given shape and dtype with all values set to the indicated "fill value"
# - `eye`, `identity`:  Create a square N-by-N identity matrix (1s on the diagonal and 0s elsewhere)

# ### Arithmetic with NumPy Arrays

# > Arrays are important because they enable you to express batch operations on data without writing any for loops. NumPy users call this vectorization. Any arithmetic operations between equal-size arrays applies the operation element-wise

# In[35]:


arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr


# NumPy array addition is elementwise.

# In[36]:


arr + arr


# NumPy array multiplication is elementwise.

# In[37]:


arr * arr


# NumPy array division is elementwise.

# In[38]:


1 / arr


# NumPy powers are elementwise, too.

# In[39]:


arr ** 0.5


# We can also raise a single value to an array!

# In[40]:


2 ** arr


# ### Basic Indexing and Slicing

# One-dimensional array index and slice the same as lists.

# In[41]:


arr = np.arange(10)
arr


# In[42]:


arr[5]


# In[43]:


arr[5:8]


# In[44]:


equiv_list = list(range(10))
equiv_list


# In[45]:


equiv_list[5:8]


# We have to jump through some hoops if we want to replace elements 5, 6, and 7 in `equiv_list` with the value 12.

# In[46]:


equiv_list[5:8] = [12] * 3
equiv_list


# However, this operation is easy with a NumPy array!

# In[47]:


arr[5:8] = 12
arr


# "Broadcasting" is the name for this behavior.
# 
# > As you can see, if you assign a scalar value to a slice, as in `arr[5:8] = 12`, the value is propagated (or broadcasted henceforth) to the entire selection. An important first distinction from Python’s built-in lists is that array slices are views on the original array. This means that the data is not copied, and any modifications to the view will be reflected in the source array.

# In[48]:


arr_slice = arr[5:8]
arr_slice


# In[49]:


x = arr_slice
x


# In[50]:


x is arr_slice


# In[51]:


y = x.copy()


# In[52]:


y is arr_slice


# In[53]:


arr_slice[1] = 12345
arr_slice


# In[54]:


arr


# The `:` slices every element in `arr_slice`.

# In[55]:


arr_slice[:] = 64
arr_slice


# In[56]:


arr


# > If you want a copy of a slice of an ndarray instead of a view, you will need to explicitly copy the array-for example, `arr[5:8].copy()`.

# In[57]:


arr_slice_2 = arr[5:8].copy()
arr_slice_2


# In[58]:


arr_slice_2[:] = 2001
arr_slice_2


# In[59]:


arr


# ### Indexing with slices

# We can slice across two or more dimensions, including the `[i, j]` notation.

# In[60]:


arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr2d


# In[61]:


arr2d[:2]


# In[62]:


arr2d[:2, 1:]


# A colon (`:`) by itself selects the entire dimension and is necessary to slice higher dimensions.

# In[63]:


arr2d[:, :1]


# In[64]:


arr2d[:2, 1:] = 0
arr2d


# Slicing multi-dimension arrays is tricky!
# ***Always check your output!***

# ### Boolean Indexing

# We can use Booleans (`True`s and `False`s) to slice arrays, too.
# Boolean indexing in Python is like combining `index()` and `match()` in Excel.

# In[65]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.random.seed(42)
data = np.random.randn(7, 4)


# In[66]:


names


# In[67]:


data


# Here `names` provides seven names for the seven rows in `data`.

# In[68]:


names == 'Bob'


# In[69]:


data[names == 'Bob']


# We can combine Boolean slicing with `:` slicing.

# In[70]:


data[names == 'Bob', 2:]


# We can use `~` to invert a Boolean.

# In[71]:


cond = names == 'Bob'
data[~cond]


# For NumPy arrays, we must use `&` and `|` instead of `and` and `or`.

# In[72]:


cond = (names == 'Bob') | (names == 'Will')
data[cond]


# We can also create a Boolean for each element.

# In[73]:


data < 0


# In[74]:


data[data < 0] = 0
data


# ## Universal Functions: Fast Element-Wise Array Functions

# > A universal function, or ufunc, is a function that performs element-wise operations on data in ndarrays. You can think of them as fast vectorized wrappers for simple functions that take one or more scalar values and produce one or more scalar results.

# In[75]:


arr = np.arange(10)


# In[76]:


np.sqrt(arr)


# Like above, we can raise a single value to a NumPy array of powers.

# In[77]:


2**arr


# `np.exp(x)` is $e^x$.

# In[78]:


np.exp(arr)


# The functions above accept one argument.
# These "unary" functions operate on one array and return a new array with the same shape.
# There are also "binary" functions that operate on two arrays and return one array.

# In[79]:


np.random.seed(42)
x = np.random.randn(8)
y = np.random.randn(8)


# In[80]:


np.maximum(x, y)


# ***Be careful!
# Function names are not the whole story.
# Check your output and read the docstring!***
# For example, `np.max()` returns the maximum of an array (instead of the elementwise maximum of two arrays for `np.maximum()`.

# In[81]:


np.max(x)


# ***Table 4-4*** from McKinney lists some fast, element-wise unary functions:
# 
# - `abs`, `fabs`: Compute the absolute value element-wise for integer, oating-point, or complex values
# - `sqrt`: Compute the square root of each element (equivalent to arr ** 0.5) 
# - `square`: Compute the square of each element (equivalent to arr ** 2)
# - `exp`: Compute the exponent $e^x$ of each element
# - `log`, `log10`, `log2`, `log1p`: Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
# - `sign`: Compute the sign of each element: 1 (positive), 0 (zero), or –1 (negative)
# - `ceil`: Compute the ceiling of each element (i.e., the smallest integer greater than or equal to thatnumber)
# - `floor`: Compute the oor of each element (i.e., the largest integer less than or equal to each element)
# - `rint`: Round elements to the nearest integer, preserving the dtype
# - `modf`: Return fractional and integral parts of array as a separate array
# - `isnan`: Return boolean array indicating whether each value is NaN (Not a Number)
# - `isfinite`, `isinf`: Return boolean array indicating whether each element is finite (non-inf, non-NaN) or infinite, respectively
# - `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh`: Regular and hyperbolic trigonometric functions
# - `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`: Inverse trigonometric functions
# - `logical_not`: Compute truth value of not x element-wise (equivalent to ~arr).

# ***Table 4-5*** from McKinney lists some fast, element-wise binary functions:
# 
# - `add`: Add corresponding elements in arrays
# - `subtract`: Subtract elements in second array from first array
# - `multiply`: Multiply array elements
# - `divide`, `floor_divide`: Divide or floor divide (truncating the remainder)
# - `power`: Raise elements in first array to powers indicated in second array
# - `maximum`, `fmax`: Element-wise maximum; `fmax` ignores `NaN`
# - `minimum`, `fmin`: Element-wise minimum; `fmin` ignores `NaN`
# - `mod`: Element-wise modulus (remainder of division)
# - `copysign`: Copy sign of values in second argument to values in first argument
# - `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`: Perform element-wise comparison, yielding boolean array (equivalent to infix operators >, >=, <, <=, ==, !=)
# - `logical_and`, `logical_or`, `logical_xor`: Compute element-wise truth value of logical operation (equivalent to infix operators & |, ^)

# ## Array-Oriented Programming with Arrays

# > Using NumPy arrays enables you to express many kinds of data processing tasks as concise array expressions that might otherwise require writing loops. This practice of replacing explicit loops with array expressions is commonly referred to as vectorization. In general, vectorized array operations will often be one or two (or more) orders of magnitude faster than their pure Python equivalents, with the biggest impact in any kind of numerical computations. Later, in Appendix A, I explain broadcasting, a powerful method for vectorizing computations.

# ### Expressing Conditional Logic as Array Operations

# > The numpy.where function is a vectorized version of the ternary expression x if condition else y.

# NumPy's `where()` is an if-else statement that operates like Excel's `if()`.

# In[82]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[83]:


np.where(cond, xarr, yarr)


# We could use a list comprehension, instead, but the list comprehension is takes longer to type, read, and troubleshoot.

# In[84]:


[(x if c else y) for x, y, c in zip(xarr, yarr, cond)]


# We could also use `np.select()`.

# In[85]:


np.select(
    condlist=[cond==True, cond==False],
    choicelist=[xarr, yarr]
)


# `np.select()` is more complex than `np.where()`, but lets us test more than two conditions and provide a default value if no condition is met.

# ### Mathematical and Statistical Methods

# > A set of mathematical functions that compute statistics about an entire array or about the data along an axis are accessible as methods of the array class. You can use aggregations (often called reductions) like sum, mean, and std (standard deviation) either by calling the array instance method or using the top-level NumPy function.
# 
# We will use these aggregations extensively in pandas.

# In[86]:


np.random.seed(42)
arr = np.random.randn(5, 4)


# In[87]:


arr.mean()


# In[88]:


arr.sum()


# The aggregation methods above aggregated the whole array.
# We can use the `axis` argument to aggregate columns (`axis=0`) and rows (`axis=1`).

# In[89]:


arr.mean(axis=1)


# In[90]:


arr[0].mean()


# In[91]:


arr.mean(axis=0)


# In[92]:


arr[:, 0].mean()


# The `.cumsum()` method returns the sum of all previous elements.

# In[93]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# We can use the `.cumsum()` method along the axis of a multi-dimension array, too.

# In[94]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr


# In[95]:


arr.cumsum(axis=0)


# In[96]:


arr.cumprod(axis=1)


# ***Table 4-6*** from McKinney lists some basic statistical methods:
# 
# - `sum`: Sum of all the elements in the array or along an axis; zero-length arrays have sum 0
# - `mean`: Arithmetic mean; zero-length arrays have NaN mean
# - `std`, `var`: Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator $n$)
# - `min`, `max`: Minimum and maximum
# - `argmin`, `argmax`: Indices of minimum and maximum elements, respectively
# - `cumsum`: Cumulative sum of elements starting from 0
# - `cumprod`: Cumulative product of elements starting from 1

# ### Methods for Boolean Arrays

# In[97]:


np.random.seed(42)
arr = np.random.randn(100)


# In[98]:


(arr > 0).sum() # Number of positive values


# In[99]:


(arr > 0).mean() # percentage of positive values


# In[100]:


bools = np.array([False, False, True, False])


# In[101]:


bools.any()


# In[102]:


bools.all()

