#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 2 - Practice (Section 3, Monday 2:45 PM)

# ## Practice

# ### Extract the year, month, and day from an integer 8-digit date (i.e., YYYYMMDD format) using `//` (integer division) and `%` (modulo division).

# Try `20080915`.

# In[1]:


ymd = 20080915


# In[2]:


ymd // 10000


# In[3]:


ymd % 10000


# In[4]:


year = ymd // 10000


# In[5]:


month = (ymd // 100) % 100


# In[6]:


day = ymd % 100


# In[7]:


(year, month, day)


# ### Use your answer above to write a function `date` that accepts an integer 8-digit date argument and returns a tuple of the year, month, and date (e.g., `return (year, month, date)`).

# In[8]:


def date(ymd):
    year = ymd // 10000
    month = (ymd // 100) % 100
    day = ymd % 100
    return (year, month, day)


# In[9]:


date(20080915)


# ### Rewrite `date` to accept an 8-digit date as an integer or string.

# In[10]:


def date(ymd):
    if type(ymd) is str:
        ymd = int(ymd)
    year = ymd // 10000
    month = (ymd // 100) % 100
    day = ymd % 100
    return (year, month, day)


# In[11]:


date(20080915)


# ### Finally, rewrite `date` to accept a list of 8-digit dates as integers or strings.

# In[12]:


def date(ymd):
    dates = []
    if type(ymd) is not list:
        ymd = [ymd]
    for i in ymd:
        if type(i) is str:
            i = int(i)
        year = i // 10000
        month = (i // 100) % 100
        day = i % 100
        dates.append((year, month, day))
    return dates


# In[13]:


date(20080915)


# In[14]:


date([20080915, '20080915'])


# ### Write a for loop that prints the squares of integers from 1 to 10.

# In[15]:


for i in range(1, 11):
    print(i**2, end=' ')


# Above, I change the `end` argument from the default '\n' to ' '.
# The default '\n' inserts a new line after value, making the output too long.

# ### Write a for loop that prints the squares of *even* integers from 1 to 10.

# In[16]:


for i in range(1, 11):
    if i % 2 == 0:
        print(i**2, end=' ')


# ### Write a for loop that sums the squares of integers from 1 to 10.

# In[17]:


total = 0
for i in range(1, 11):
    total += i**2


# In[18]:


total


# Above, I use `total += i`, which is equivalent to `total = total + i`.

# ### Write a for loop that sums the squares of integers from 1 to 10 but stops before the sum exceeds 50.

# In[19]:


total = 0
for i in range(1, 11):
    if (total + i**2) > 50:
        break
    total += i**2


# In[20]:


total


# ### FizzBuzz 

# Write a for loop that prints the numbers from 1 to 100. 
# For multiples of three print "Fizz" instead of the number.
# For multiples of five print "Buzz". 
# For numbers that are multiples of both three and five print "FizzBuzz".
# More [here](https://blog.codinghorror.com/why-cant-programmers-program/).

# In[21]:


for i in range(1, 101):
    is_mult_3 = (i % 3 == 0)
    is_mult_5 = (i % 5 == 0)
    if is_mult_3 and is_mult_5:
        print('FizzBuzz', end=' ')
    elif is_mult_3:
        print('Fizz', end=' ')
    elif is_mult_5:
        print('Buzz', end=' ')
    else:
        print(i, end=' ')


# ### Use ternary expressions to make your FizzBuzz code more compact.

# In[22]:


for i in range(1, 101):
    is_mult_3 = (i % 3 == 0)
    is_mult_5 = (i % 5 == 0)
    print('Fizz'*is_mult_3 + 'Buzz'*is_mult_5 if is_mult_3 or is_mult_5 else i, end=' ')


# The solution above is shorter and uses from neat tricks, but I consider the previous solution easier to read and troubleshoot.

# ### Triangle

# Write a function `triangle` that accepts a positive integer $N$ and prints a numerical triangle of height $N-1$.
# For example, `triangle(N=6)` should print:
# 
# ```
# 1
# 22
# 333
# 4444
# 55555
# ```

# In[23]:


def triangle(N):
    for i in range(1, N):
        print(str(i) * i)


# In[24]:


triangle(6)


# The solution above works because a multiplying a string by `i` concatenates `i` copies of the string.

# In[25]:


'Test' + 'Test' + 'Test'


# In[26]:


'Test' * 3


# ### Two Sum

# Write a function `two_sum` that does the following.
# 
# Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to target.
# 
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# 
# You can return the answer in any order.
# 
# Here are some examples:
# 
# Example 1:
# 
# Input: `nums = [2,7,11,15]`, `target = 9` \
# Output: `[0,1]` \
# Explanation: Because `nums[0] + nums[1] == 9`, we return `[0, 1]`.
# 
# Example 2:
# 
# Input: `nums = [3,2,4]`, `target = 6` \
# Output: `[1,2]` \
# 
# Example 3:
# 
# Input: `nums = [3,3]`, `target = 6` \
# Output: `[0,1]` \
# 
# I saw this question on [LeetCode](https://leetcode.com/problems/two-sum/description/).

# In[27]:


def two_sum(nums, target):
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] + nums[j] == target:
                return [j, i]


# In[28]:


two_sum(nums = [2,7,11,15], target = 9)


# In[29]:


two_sum(nums = [3,2,4], target = 6)


# In[30]:


two_sum(nums = [3,3], target = 6)


# We can write more efficient code once we learn other data structures in chapter 3 of McKinney!

# ### Best Time

# Write a function `best_time` that solves the following.
# 
# You are given an array `prices` where `prices[i]` is the price of a given stock on the $i^{th}$ day.
# 
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# 
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
# 
# Here are some examples:
# 
# Example 1:
# 
# Input: `prices = [7,1,5,3,6,4]` \
# Output: `5` \
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
# 
# Example 2:
# 
# Input: `prices = [7,6,4,3,1]` \
# Output: `0` \
# Explanation: In this case, no transactions are done and the max profit = 0.
# 
# I saw this question on [LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/).

# In[31]:


def max_profit(prices):
        min_price = prices[0]
        max_profit = 0
        for price in prices:
            min_price = price if price < min_price else min_price
            profit = price - min_price
            max_profit = profit if profit > max_profit else max_profit
        return max_profit


# In[32]:


max_profit(prices=[7,1,5,3,6,4])


# In[33]:


max_profit(prices=[7,6,4,3,1])

