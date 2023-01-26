#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 3 - Practice (Section 2, Wednesday 2:45 PM)

# ## Practice

# ### Swap the values assigned to `a` and `b` using a third variable `c`.

# In[1]:


a = 1


# In[2]:


b = 2


# In[3]:


print(f'a is {a} and b is {b}')


# In[4]:


c = a


# In[5]:


c == a


# In[6]:


c is a


# In[7]:


a = b


# In[8]:


b = c


# In[9]:


print(f'a is {a} and b is {b}')


# In[10]:


get_ipython().run_line_magic('who', '')


# In[11]:


del c


# In[12]:


get_ipython().run_line_magic('who', '')


# ### Swap the values assigned to `a` and `b` ***without*** using a third variable `c`.

# In[13]:


a = 1


# In[14]:


b = 2


# In[15]:


print(f'a is {a} and b is {b}')


# In[16]:


type((a, b))


# In[17]:


b, a = a, b


# In[18]:


print(f'a is {a} and b is {b}')


# ---
# Here is a trap:

# In[19]:


d = a, b


# In[20]:


type(d)


# In[21]:


e, f = d


# In[22]:


print(f'e is {e} and f is {f}')


# In[23]:


# ValueError: not enough values to unpack (expected 3, got 2)
# g, h, i = d


# ---

# ### What is the output of the following code and why?

# In[24]:


1 == (1, 1, 1)


# In[25]:


1, 1, 1 == (1, 1, 1)


# In[26]:


1, 1 == (1, 1, 1)


# In[27]:


(1, 1, 1) == (1, 1, 1)


# In[28]:


(1, 1, 1) == 1, 1, 1


# In[29]:


1, 1, 1 == 1, 1, 1


# ### Create a list `l1` of integers from 1 to 100.

# In[30]:


l1 = list(range(1, 101))
print(l1)


# In[31]:


l1[:5] 


# In[32]:


l1[5:10]


# ### Slice `l1` to create a list of integers from 60 to 50 (inclusive).

# Name this list `l2`.

# In[33]:


l2 = l1[49:60]
l2.sort(reverse=True)
print(l2)


# In[34]:


l1[59:48:-1]


# In[35]:


l1[49:60][::-1]


# In[36]:


l2 = l1[49:60]
l2.reverse()
l2


# ### Create a list `l3` of odd integers from 1 to 21.

# In[37]:


l3 = list(range(1, 22, 2))
l3


# ### Create a list `l4` of the squares of integers from 1 to 100.

# In[38]:


get_ipython().run_cell_magic('timeit', '', 'l4 = []\nfor i in range(1, 101):\n    l4.append(i**2)')


# A list comprehension is easier to write and read and considered "more Pythonic".
# Here is the pseudo code for a list comprehension:
# ```
# [f(x) for x in some range if some condition]
# ```

# In[39]:


get_ipython().run_cell_magic('timeit', '', 'l4 = [x**2 for x in range(1, 101)]')


# ### Create a list `l5` that contains the squares of ***odd*** integers from 1 to 100.

# In[40]:


l5 = [x**2 for x in range(1, 101, 2)]
print(l5)


# Here is a less efficient solution that uses the `if some condition` part of list comprehensions:

# In[41]:


print([x**2 for x in range(1, 101) if x%2 != 0])


# ### Use a lambda function to sort `strings` by the last letter.

# In[42]:


strings = ['card', 'aaaa', 'foo', 'bar', 'abab']


# In[43]:


strings.sort(key=lambda x: x[-1])


# In[44]:


strings


# ### Given an integer array `nums` and an integer `k`, return the $k^{th}$ largest element in the array.

# Note that it is the $k^{th}$ largest element in the sorted order, not the $k^{th}$ distinct element.
# 
# Example 1:
# 
# Input: `nums = [3,2,1,5,6,4]`, `k = 2` \
# Output: `5`
# 
# Example 2:
# 
# Input: `nums = [3,2,3,1,2,4,5,5,6]`, `k = 4` \
# Output: `4`
# 
# I saw this question on [LeetCode](https://leetcode.com/problems/kth-largest-element-in-an-array/).

# In[45]:


def nums(x, k):
    x_copy = x.copy()
    x_copy.sort()
    return x_copy[-k]


# In[46]:


nums(x=[3,2,1,5,6,4], k=2)


# ### Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. 

# You may return the answer in any order.
# 
# Example 1:
# 
# Input: `nums = [1,1,1,2,2,3]`, `k = 2` \
# Output: `[1,2]`
# 
# Example 2:
# 
# Input: `nums = [1]`, `k = 1` \
# Output: `[1]`
# 
# I saw this question on [LeetCode](https://leetcode.com/problems/top-k-frequent-elements/).

# In[47]:


def nums(nums, k):
    counts = {}
    for n in nums:
        if n in counts:
            counts[n] += 1
        else:
            counts[n] = 1
    return [x[0] for x in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]]


# In[48]:


nums(nums=[1,1,1,2,2,3], k=2)


# ### Test whether the given strings are palindromes.

# Input: `["aba", "no"]` \
# Output: `[True, False]`

# In[49]:


def is_palindrome(x):
    return [list(y) == list(y)[::-1] for y in x]


# In[50]:


is_palindrome(["aba", "no"])


# ### Write a function `returns()` that accepts lists of prices and dividends and returns a list of returns.

# In[51]:


prices = [100, 150, 100, 50, 100, 150, 100, 150]
dividends = [1, 1, 1, 1, 2, 2, 2, 2]


# In[52]:


def returns(p, d):
    r = [None]
    for t in range(1, len(p)):
        pt = p[t]
        ptm1 = p[t-1]
        dt = d[t]
        rt = (pt - ptm1 + dt) / ptm1
        r.append(rt)
        
    return r


# In[53]:


returns(p=prices, d=dividends)


# ### Rewrite the function `returns()` so it returns lists of returns, capital gains yields, and dividend yields.

# In[54]:


def returns(p, d):
    r, cg, dy = [None], [None], [None]
    for t in range(1, len(p)):
        pt = p[t]
        ptm1 = p[t-1]
        dt = d[t]
        cgt = (pt - ptm1) / ptm1
        dyt = dt / ptm1
        rt = cgt + dyt
        r.append(rt)
        cg.append(cgt)
        dy.append(dyt)
        
    return {'r': r, 'cg': cg, 'dy': dy}


# In[55]:


returns(p=prices, d=dividends)


# ### Rescale and shift numbers so that they cover the range `[0, 1]`.

# Input: `[18.5, 17.0, 18.0, 19.0, 18.0]` \
# Output: `[0.75, 0.0, 0.5, 1.0, 0.5]`

# In[56]:


numbers = [18.5, 17.0, 18.0, 19.0, 18.0]


# In[57]:


def rescale(x):
    x_min = min(x)
    x_max = max(x)
    return [(x - x_min) / (x_max - x_min) for x in x]


# In[58]:


rescale(numbers)

