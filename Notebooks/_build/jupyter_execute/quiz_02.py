#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Initialize Otter
import otter
grader = otter.Notebook("quiz_02.ipynb")


# # Quiz 2

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
#     1. Public tests check your answers for correct types and shapes but may not completely check your answers
#     1. Hidden tests completely check your answers but will not be available until after the due date
# 1. You may ask technical questions on Canvas Discussions, but the quiz is an individual effort.

# ## Packages and Settings

# In[ ]:


import numpy as np
get_ipython().run_line_magic('precision', '4')


# ## Questions

# ### Question 1

# Write a function `npv()` that calculates the net present value of a NumPy array of cashflows given a discount rate.
# Assume the following:
# 
# 1. Argument `c` is a NumPy array of annual cash flows, starting at $t=0$
# 1. Argument `r` is a float of the annual discount rate as a decimal (i.e., `r=0.1` indicates $r=10\%$)
# 
# ***Note: This question has 1 hidden test worth a total of 20 points.***
# 
# _Points:_ 50

# In[ ]:


def npv(c, r):
    ...


# In[ ]:


grader.check("q1")


# ### Question 2

# Write a function `totret()` that calculates the total return of a NumPy array of returns.
# Assume the following:
# 
# 1. Argument `r` is a NumPy array of decimal returns (i.e., `r`=0.1 indicates $r=10\%$ and $r > -100\%$ for all $r$)
# 1. Function `totret()` should return total return as a decimal
# 
# ***Note: This question has 2 hidden tests worth a total of 20 points.***
# 
# _Points:_ 50

# In[ ]:


def totret(r):
    ...


# In[ ]:


grader.check("q2")


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[ ]:


# Save your notebook first, then run this cell to export your submission.
grader.export(pdf=False, run_tests=True)


#  
