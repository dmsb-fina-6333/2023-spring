#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Initialize Otter
import otter
grader = otter.Notebook("quiz_01.ipynb")


# # Quiz 1

# ## Instructions
# 
# 1. After you answer a question, you should run its public tests.
# 1. After you answer every question, you should:
#     1. Restart your kernel and clear all output
#     1. Run up to the last cell
#     1. Save your notebook
#     1. Run the last cell to create a .zip file for Gradescope
#     1. Upload this .zip file to Gradescope
#     1. ***Make sure your local autograder results match your Gradescope autograder results***
# 1. This quiz has public and hidden tests:
#     1. Public tests check if your answers are the correct types and shapes, but they may not check if your answers are exactly correct
#     1. Hidden tests check if your answers are exactly correct, but their results are not available until after the due date
# 1. You may ask technical questions on Canvas Discussions, but the quiz is an individual effort.

# ## Questions

# ### Question 1

# Create a list named `list_1` that contains the squares for integers from 1 to 100.
# The first and last values in `list_1` should be $1^2 = 1$ and $100^2 = 10,000$, respectively.
# 
# _Points:_ 30

# In[ ]:


...


# In[ ]:


grader.check("q1")


# ### Question 2

# Create an integer `int_2` that contains the sum of squared integers from 1 to 100 inclusive.
# This `int_2` contains the sum of the values in `list_1` from question 1.
# 
# _Points:_ 30

# In[ ]:


...


# In[ ]:


grader.check("q2")


# ### Question 3

# Write a function `fun_3` that sums that squares of integers between its arguments `start` and `stop` inclusive.
# If `start=1` and `stop=2`, `fun_3` should return $1^2 + 2^2 = 1 + 4 = 5$.
# If `start=1` and `stop=100`, `fun_3` should return your answer from question 2.
# I will test your function at several values of `start` and `stop`, you may assume that `stop` is always greater than `start`.
# 
# ***Note:***
# This question has 1 hidden test worth 10 points.
# The autograder will not show you hidden test results until after the due date, and Gradescope will show your quiz score as `~/100` until after the due date.
# 
# _Points:_ 40

# In[ ]:


def fun_3(start, stop):


# In[ ]:


grader.check("q3")


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[ ]:


# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False, run_tests=True)


#  
