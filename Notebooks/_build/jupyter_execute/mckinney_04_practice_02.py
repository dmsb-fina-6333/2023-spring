#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 4 - Practice (Section 2, Wednesday 2:45 PM)

# In[1]:


import numpy as np
get_ipython().run_line_magic('precision', '4')


# ## Practice

# ### Create a 1-dimensional array named `a1` that counts from 0 to 24 by 1.

# In[2]:


np.array(range(25))


# In[3]:


np.arange(25)


# In[4]:


a1 = np.arange(25)


# ### Create a 1-dimentional array named `a2` that counts from 0 to 24 by 3.

# In[5]:


a2 = np.arange(0, 25, 3)
a2


# ### Create a 1-dimentional array named `a3` that counts from 0 to 100 by multiples of 3 or 5.

# In[6]:


np.array([i for i in range(101) if (i%3==0) or (i%5==0)])


# In[7]:


a3 = np.arange(101)
a3 = a3[(a3%3==0) | (a3%5==0)] # NumPy only accepts & for "and" and | for "or"


# ### Create a 1-dimensional array `a3` that contains the squares of the even integers through 100,000.

# How much faster is the NumPy version than the list comprehension version?

# In[8]:


np.array([i**2 for i in range(100_001) if (i%2==0)])


# In[9]:


np.array([i**2 for i in range(0, 100_001, 2)])


# In[10]:


np.arange(0, 100_001, 2, dtype=np.int64)**2


# ---

# ***Note:***
# On some platforms, `np.arange(0, 100_001, 2)` returns an array of 32-bit integers.
# If we square this array of 32-bit integers, we get the wrong answer because the large values (e.g., $100,000^2$) are too large to represent as 32-bit integers.
# Since we know that we need 54-bit integers for this calculation, we should explcitly set either `dtype='int64'` or `dtype=np.int64`.

# In[11]:


np.arange(0, 100_001, 2, dtype='int64')**2


# In[12]:


np.arange(0, 100_001, 2, dtype=np.int64)**2


# This [StackOverflow answer](https://stackoverflow.com/a/1970697/334755) is this best explanation I have found of this behavior.

# ---

# ### Write a function that mimic Excel's `pv` function.

# Here is how we call Excel's `pv` function:
# `=PV(rate, nper, pmt, [fv], [type])`
# We can use the annuity and lump sum present value formulas.
# 
# Present value of an annuity payment `pmt`:
# $PV_{pmt} = \frac{pmt}{rate} \times \left(1 - \frac{1}{(1+rate)^{nper}} \right)$
# 
# Present value of a lump sum `fv`:
# $PV_{fv} = \frac{fv}{(1+rate)^{nper}}$

# In[13]:


def pv(rate, nper, pmt=None, fv=None, type = 'END'):
    if pmt is not None: # calculate PV of pmt, if given
        pv_pmt = (pmt / rate) * (1 - (1 + rate)**(-nper))
    else:
        pv_pmt = 0

    if fv is not None: # calculate PV of fv, if given
        pv_fv = fv / (1 + rate)**nper
    else:
        pv_fv = 0
    
    if type=='END': # as-is if end of period payments
        return -1 * (pv_pmt + pv_fv)
    elif type=='BGN': # undo one period of discounting if bgn of period payments
        return -1 * (pv_pmt + pv_fv) * (1 + rate)
    else: # otherwise, ask use to specify end or bgn of period payments
        print('Please enter END or BGN for the type argument')


# In[14]:


pv(rate=0.1, nper=10, pmt=100, fv=1000, type='END')


# ### Write a function that mimic Excel's `fv` function.

# In[15]:


def fv(rate, nper, pmt=None, pv=None, type = 'END'):
    if pmt is not None: # calculate PV of pmt, if given
        fv_pmt = (pmt / rate) * ((1 + rate)**nper - 1)
    else:
        fv_pmt = 0

    if fv is not None: # calculate PV of fv, if given
        fv_pv = pv * (1 + rate)**nper
    else:
        fv_fv = 0
    
    if type=='END': # as-is if end of period payments
        return -1 * (fv_pmt + fv_pv)
    elif type=='BGN': # undo one period of discounting if bgn of period payments
        return -1 * (fv_pmt + fv_pv) * (1 + rate)
    else: # otherwise, ask use to specify end or bgn of period payments
        print('Please enter END or BGN for the type argument')


# In[16]:


fv(rate=0.1, nper=10, pmt=100, pv=-1000, type='END')


# ### Replace the negative values in `data` with -1 and positive values with +1.

# In[17]:


np.random.seed(42)
data = np.random.randn(7, 7)
data


# In[18]:


data_c = data.copy()
data_c[data_c < 0] = -1
data_c[data_c > 0] = +1
data_c


# In[19]:


np.where(data < 0, -1, np.where(data > 0, +1, data))


# In[20]:


np.allclose(np.where(data < 0, -1, np.where(data > 0, +1, data)), data_c)


# In[21]:


np.select(
    condlist=[data<0, data>0],
    choicelist=[-1, +1],
    default=data
)


# ### Write a function `npmts()` that calculates the number of payments that generate $x\%$ of the present value of a perpetuity.

# Your `npmts()` should accept arguments `c1`, `r`, and `g` that represent  $C_1$, $r$, and $g$.
# The present value of a growing perpetuity is $PV = \frac{C_1}{r - g}$, and the present value of a growing annuity is $PV = \frac{C_1}{r - g}\left[ 1 - \left( \frac{1 + g}{1 + r} \right)^t \right]$.

# We can use the growing annuity and perpetuity formulas to show: $x = \left[ 1 - \left( \frac{1 + g}{1 + r} \right)^t \right]$. 
# 
# Then: $1 - x = \left( \frac{1 + g}{1 + r} \right)^t$.
# 
# Finally: $t = \frac{\log(1-x)}{\log\left(\frac{1 + g}{1 + r}\right)}$
# 
# ***We do not need to accept an argument `c1` because $C_1$ cancels out!***

# In[22]:


def npmts(x, r, g):
    return np.log(1-x) / np.log((1 + g) / (1 + r))


# In[23]:


npmts(0.5, 0.1, 0.05)


# ### Write a function that calculates the internal rate of return given a NumPy array of cash flows.

# Here are some data where the $IRR$ is obvious!

# In[24]:


c = np.array([-100, 110])
irr = 0.1


# We want to replicate the following calculation with NumPy:

# In[25]:


c[0]/(1+irr)**0 + c[1]/(1+irr)**1


# The following NumPy code calculates the present value interest factor for each cash flow:

# In[26]:


1 / (1 + irr)**np.arange(len(c)) # present value interest factor


# The following NumPy code calculates the present value for each cash flow:

# In[27]:


c / (1 + irr)**np.arange(len(c)) # present value of each cash flow


# We sum these present values of each cash flow to calculate the net present value:

# In[28]:


(c / (1 + irr)**np.arange(len(c))).sum()


# The $IRR$ is the discount rate where $NPV=0$.
# We can can use the NumPy code above to try different discount rates until $NPV=0$.
# The following code is crude, but sufficient for week three of our class and highlights two tools:
# 
# 1. Using a `while` to try different discount rates until our answer is within some tolerance
# 1. Using NumPy to perform repetitive calculations without a `for` loop

# In[29]:


def irr(c, guess=0, tol=0.0001, inc=0.0001):
    npv = 42
    irr = guess
    while np.abs(npv) > tol:
        irr += inc
        npv = (c / (1 + irr)**np.arange(len(c))).sum()
        
    return irr


# In[30]:


c = np.array([-100, 110])


# In[31]:


irr(c)


# ### Write a function `returns()` that accepts *NumPy arrays* of prices and dividends and returns a *NumPy array* of returns.

# In[32]:


prices = np.array([100, 150, 100, 50, 100, 150, 100, 150])
dividends = np.array([1, 1, 1, 1, 2, 2, 2, 2])


# In[33]:


def returns(p, d):
    return (p[1:] - p[:-1] + d[1:]) / p[:-1]


# In[34]:


returns(p=prices, d=dividends)


# ### Rewrite the function `returns()` so it returns *NumPy arrays* of returns, capital gains yields, and dividend yields.

# In[35]:


def returns(p, d):
    cg = (p[1:] - p[:-1]) / p[:-1]
    dy = d[1:] / p[:-1]
    r = cg + dy
    return {'r': r, 'cg': cg, 'dy': dy}


# In[36]:


returns(p=prices, d=dividends)


# ### Rescale and shift numbers so that they cover the range [0, 1]

# Input: `np.array([18.5, 17.0, 18.0, 19.0, 18.0])` \
# Output: `np.array([0.75, 0.0, 0.5, 1.0, 0.5])`

# In[37]:


numbers = np.array([18.5, 17.0, 18.0, 19.0, 18.0])


# In[38]:


(numbers - numbers.min()) / (numbers.max() - numbers.min())


# ### Write functions `var()` and `std()` that calculate variance and standard deviation.

# NumPy's `.var()` and `.std()` methods return *population* statistics (i.e., denominators of $n$).
# The pandas equivalents return *sample* statistics (denominators of $n-1$), which are more appropriate for financial data analysis where we have a sample instead of a population.
# 
# 
# Both function should have an argument `sample` that is `True` by default so both functions return sample statistics by default.
# 
# Use the output of `returns()` to compare your functions with NumPy's `.var()` and `.std()` methods.

# In[39]:


np.random.seed(42)
r = np.random.randn(1_000_000)
r


# In[40]:


def var(x, ddof=0):
    N = len(x)
    return ((x - x.mean())**2).sum() / (N - ddof)


# In[41]:


r.var()


# In[42]:


var(r)


# In[43]:


np.allclose(r.var(), var(r))


# In[44]:


np.allclose(r.var(ddof=1), var(r, ddof=1))


# In[45]:


def std(x, ddof=0):
    return np.sqrt(var(x=x, ddof=ddof))


# In[46]:


r.std()


# In[47]:


std(r)


# In[48]:


np.allclose(r.std(), std(r))


# In[49]:


np.allclose(r.std(ddof=1), std(r, ddof=1))

