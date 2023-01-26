#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 3 - Practice (Blank)

# ## Practice

# ### Swap the values assigned to `a` and `b` using a third variable `c`.

# In[1]:


a = 1


# In[2]:


b = 2


# ### Swap the values assigned to `a` and `b` ***without*** using a third variable `c`.

# In[3]:


a = 1


# In[4]:


b = 2


# ### What is the output of the following code and why?

# In[5]:


1, 1, 1 == (1, 1, 1)


# ### Create a list `l1` of integers from 1 to 100.

# ### Slice `l1` to create a list of integers from 60 to 50 (inclusive).

# Name this list `l2`.

# ### Create a list `l3` of odd integers from 1 to 21.

# ### Create a list `l4` of the squares of integers from 1 to 100.

# ### Create a list `l5` that contains the squares of ***odd*** integers from 1 to 100.

# ### Use a lambda function to sort `strings` by the last letter.

# In[6]:


strings = ['card', 'aaaa', 'foo', 'bar', 'abab']


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

# ### Test whether the given strings are palindromes.

# Input: `["aba", "no"]` \
# Output: `[True, False]`

# ### Write a function `returns()` that accepts lists of prices and dividends and returns a list of returns.

# In[7]:


prices = [100, 150, 100, 50, 100, 150, 100, 150]
dividends = [1, 1, 1, 1, 2, 2, 2, 2]


# ### Rewrite the function `returns()` so it returns lists of returns, capital gains yields, and dividend yields.

# ### Rescale and shift numbers so that they cover the range `[0, 1]`.

# Input: `[18.5, 17.0, 18.0, 19.0, 18.0]` \
# Output: `[0.75, 0.0, 0.5, 1.0, 0.5]`

# In[8]:


numbers = [18.5, 17.0, 18.0, 19.0, 18.0]

