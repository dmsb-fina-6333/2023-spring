#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 2 - Practice (Blank)

# ## Practice

# ### Extract the year, month, and day from an integer 8-digit date (i.e., YYYYMMDD format) using `//` (integer division) and `%` (modulo division).

# Try `20080915`.

# ### Use your answer above to write a function `date` that accepts an integer 8-digit date argument and returns a tuple of the year, month, and date (e.g., `return (year, month, date)`).

# ### Rewrite `date` to accept an 8-digit date as an integer or string.

# ### Finally, rewrite `date` to accept a list of 8-digit dates as integers or strings.

# Return a list of tuples of year, month, and date.

# ### Write a for loop that prints the squares of integers from 1 to 10.

# ### Write a for loop that prints the squares of *even* integers from 1 to 10.

# ### Write a for loop that sums the squares of integers from 1 to 10.

# ### Write a for loop that sums the squares of integers from 1 to 10 but stops before the sum exceeds 50.

# ### FizzBuzz

# Write a for loop that prints the numbers from 1 to 100. 
# For multiples of three print "Fizz" instead of the number.
# For multiples of five print "Buzz". 
# For numbers that are multiples of both three and five print "FizzBuzz".
# More [here](https://blog.codinghorror.com/why-cant-programmers-program/).

# ### Use ternary expressions to make your FizzBuzz code more compact.

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
