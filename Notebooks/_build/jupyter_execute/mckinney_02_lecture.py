#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 2 - Python Language Basics, IPython, and Jupyter Notebooks

# ## Introduction

# We must understand the basics of Python before we can use it to analyze financial data.
# Chapter 2 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) provides a crash course in Python's syntax, and chapter 3 provides a crash course in Python's built-in data structures.
# This notebook focuses on the "Python Language Basics" in section 2.3, which covers language semantics, scalar types, and control flow.
# 
# ***Note:*** 
# Indented block quotes are from McKinney unless otherwise indicated.
# The section numbers here differ from McKinney because we will only discuss some topics.

# ## Language Semantics

# ### Indentation, not braces

# > Python uses whitespace (tabs or spaces) to structure code instead of using braces as in many other languages like R, C++, Java, and Perl.
# 
# ***So, spaces are more than cosmetic in Python.***
# For non-Python programmers, white space is often Python's defining feature.
# Here is a for loop with an if block that shows how Python uses white space.

# In[1]:


array = [1, 2, 3]
pivot = 2
less = []
greater = []

for x in array:
    if x < pivot:
        print(f'{x} is less than {pivot}')
        less.append(x)        
    else:
        print(f'{x} is NOT less than {pivot}')
        greater.append(x)


# In[2]:


less


# In[3]:


greater


# ***Note:***
# We will use f-string print statements wherever we can.
# These f-string print statements are easy to use, and I do not want to teach old approaches when the new ones are better.

# ### Comments

# > Any text preceded by the hash mark (pound sign) # is ignored by the Python interpreter. This is often used to add comments to code. At times you may also want to exclude certain blocks of code without deleting them.
# 
# The Python interpreter ignores any code after a hash mark `#` on a given line.
# We can quickly comment/un-comment lines of code with the `<Ctrl>-/` shortcut.

# In[4]:


# 5 + 5


# ### Function and object method calls

# > You call functions using parentheses and passing zero or more arguments, optionally assigning the returned value to a variable:
# > ```python
# >     result = f(x, y, z)
# >     g()
# > ```
# > Almost every object in Python has attached functions, known as methods, that have access to the object's internal contents. You can call them using the following > syntax:
# > ```python
# >     obj.some_method(x, y, z)
# > ```
# > Functions can take both positional and keyword arguments:
# > ```python
# >     result = f(a, b, c, d=5, e='foo')
# > ```
# > More on this later.
# 
# We can write a function that adds two numbers.

# In[5]:


def add_numbers(a, b):
    return a + b


# In[6]:


add_numbers(5, 5)


# We can write a function that adds two strings separated by a space.

# In[7]:


def add_strings(a, b):
    return a + ' ' + b


# In[8]:


add_strings('5', '5')


# ***What is the difference between `print()` and `return`?***
# `print()` returns its arguments to the console or "standard output", whereas `return` returns its argument as an output we can assign to variables.
# In the example below, we use the `return` line to assign the output of `add_string_2()` to the variable `return_from_add_strings_2`.
# The `print()` line prints to the console or "standard output", but its output is not assigned or captured.

# In[9]:


def add_strings_2(a, b):
    string_to_print = a + ' ' + b + ' (this is from the print statement)'
    string_to_return = a + ' ' + b + ' (this is from the return statement)'
    print(string_to_print)
    return string_to_return


# In[10]:


returned = add_strings_2('5', '5')


# In[11]:


returned


# ### Variables and argument passing

# > When assigning a variable (or name) in Python, you are creating a reference to the object on the righthand side of the equals sign.

# In[12]:


a = [1, 2, 3]


# In[13]:


a


# If we assign `a` to a new variable `b`, both `a` and `b` refer to the *same* object.
# This same object is the list `[1, 2, 3]`.
# If we change `a`, we also change `b`, because these variables or names refer to the *same* object.

# In[14]:


b = a


# In[15]:


b


# ***Variables `a` and `b` refer to the same object, a list `[1, 2, 3]`.***
# We will learn more about lists (and tuples and dictionaries) in chapter 3 of McKinney.

# In[16]:


a is b


# ***If we modify `a` by appending a 4, we change `b` because `a` and `b` refer to the same list.***

# In[17]:


a.append(4)


# In[18]:


a


# In[19]:


b


# ***Likewise, if we modify `b` by appending a 5, we change `a`, too!***

# In[20]:


b.append(5)


# In[21]:


a


# In[22]:


b


# The behavior is useful but a double-edged sword!
# [Here](https://nedbatchelder.com/text/names.html) is a deeper discussion of this behavior.

# ### Dynamic references, strong types

# > In contrast with many compiled languages, such as Java and C++, object references in Python have no type associated with them.
# 
# In Python, 
# 
# 1. We do not declare variables and their types
# 1. We can change variables' types because variables are only names that refer to objects
# 
# *Dynamic references* mean we can reassign a variable to a new object in Python.
# For example, we can reassign `a` from a list to an integer to a string.

# In[23]:


a


# In[24]:


type(a)


# In[25]:


a = 5
type(a)


# In[26]:


a = 'foo'
type(a)


# *Strong types* mean Python typically will not convert object types.
# For example, the code  returns either `'55'` as a string or `10` as an integer in many programming languages.
# However, `'5' + 5` returns an error in Python.

# In[27]:


# '5' + 5


# However, Python implicitly converts integers to floats.

# In[28]:


a = 4.5
b = 2
print(f'a is {type(a)}, b is {type(b)}')
a / b


# In the previous code cell:
# 
# 1. The 'a is ...' output prints because of the explicit `print()` function call
# 1. The output of `a / b` prints (or displays) because it is the last line in the code cell

# If we want integer division (or floor division), we have to use `//`.

# In[29]:


5 // 2


# In[30]:


5 / 2


# ### Attributes and methods

# We can use tab completion to list attributes (characteristics stored inside objects) and methods (functions associated with objects).

# In[31]:


a = 'foo'


# In[32]:


a.capitalize()


# ### Imports

# > In Python a module is simply a file with the .py extension containing Python code.
# 
# We can import with `import` statements, which have several syntaxes.
# The basic syntax uses the module name as the prefix to separate module items from our current namespace.

# In[33]:


import pandas


# The `import as` syntax lets us define an abbreviated prefix.

# In[34]:


import pandas as pd


# We can also import one or more items from a package into our namespace with the following syntaxes.

# In[35]:


from pandas import DataFrame


# In[36]:


from pandas import DataFrame as df


# ### Binary operators and comparisons

# Binary operators work like Excel.

# In[37]:


5 - 7


# In[38]:


12 + 21.5


# In[39]:


5 <= 2


# We can operate during an assignment to avoid two names referring to the same object.

# In[40]:


a = [1, 2, 3]
b = a
c = list(a)


# In[41]:


a is b


# In[42]:


a is c


# Here `a` and `c` have the same *values* but are not the same object!

# In[43]:


a == c


# In[44]:


a is not c


# In Python, `=` is the assignment operator, `==` tests equality, and `!=` tests inequality.

# In[45]:


a == c


# In[46]:


a != c


# `a` and `c` have the same values but reference different objects in memory.

# ***Table 2-1*** from McKinney summarizes the binary operators.
# 
# - `a + b` : Add a and b
# - `a - b` : Subtract b from a
# - `a * b` : Multiply a by b
# - `a / b` : Divide a by b
# - `a // b` : Floor-divide a by b, dropping any fractional remainder
# - `a ** b` : Raise a to the b power
# - `a & b` : True if both a and b are True; for integers, take the bitwise AND
# - `a | b` : True if either a or b is True; for integers, take the bitwise OR
# - `a ^ b` : For booleans, True if a or b is True , but not both; for integers, take the bitwise EXCLUSIVE-OR
# - `a == b` : True if a equals b
# - `a != b`: True if a is not equal to b
# - `a <= b, a < b` : True if a is less than (less than or equal) to b
# - `a > b, a >= b`: True if a is greater than (greater than or equal) to b
# - `a is b` : True if a and b reference the same Python object
# - `a is not b` : True if a and b reference different Python objects

# ### Mutable and immutable objects

# > Most objects in Python, such as lists, dicts, NumPy arrays, and most user-defined
# types (classes), are mutable. This means that the object or values that they contain can
# be modified.
# 
# Lists are mutable, so we can modify them.

# In[47]:


a_list = ['foo', 2, [4, 5]]


# ***Python is zero-indexed! The first element has a zero subscript `[0]`!***

# In[48]:


a_list[0]


# In[49]:


a_list[2]


# In[50]:


a_list[2][0]


# In[51]:


a_list[2] = (3, 4)


# Tuples are *immutable*, so we cannot modify them.

# In[52]:


a_tuple = (3, 5, (4, 5))


# The Python interpreter returns an error if we try to modify `a_tuple` because tuples are immutable.

# In[53]:


# a_tuple[1] = 'four'


# ***Note:***
# Tuples do not require `()`, but `()` improve readability.

# In[54]:


test = 1, 2, 3


# In[55]:


type(test)


# We will learn more about Python's built-in data structures in McKinney chapter 3.

# ## Scalar Types

# > Python along with its standard library has a small set of built-in types for handling numerical data, strings, boolean ( True or False ) values, and dates and time. These "single value" types are sometimes called scalar types and we refer to them in this book as scalars. See Table 2-4 for a list of the main scalar types. Date and time handling will be discussed separately, as these are provided by the datetime module in the standard  library.
# 
# ***Table 2-2*** from McKinney lists the standard scalar types.
# 
# - `None`: The Python "null" value (only one instance of the None object exists)
# - `str`: String type; holds Unicode (UTF-8 encoded) strings
# - `bytes`: Raw ASCII bytes (or Unicode encoded as bytes)
# - `float`: Double-precision (64-bit) floating-point number (note there is no separate double type)
# - `bool`: A True or False value
# - `int`: Arbitrary precision signed integer

# ### Numeric types

# In Python, integers are unbounded, and `**` raises numbers to a power.
# So, `ival ** 6` is $17239781^6$.

# In[56]:


ival = 17239871
ival ** 6


# Floats (decimal numbers) are 64-bit in Python.

# In[57]:


fval = 7.243


# In[58]:


type(fval)


# Dividing integers yields a float, if necessary.

# In[59]:


3 / 2


# We have to use `//` if we want C-style integer division (i.e., $3 / 2 = 1$).

# In[60]:


3 // 2


# ### Booleans

# > The two Boolean values in Python are written as True and False. Comparisons and other conditional expressions evaluate to either True or False. Boolean values are combined with the and and or keywords.
# 
# Python is case sensitive, so we must type Booleans as `True` and `False`.

# In[61]:


True and True


# In[62]:


(5 > 1) and (10 > 5)


# In[63]:


False or True


# In[64]:


(5 > 1) or (10 > 5)


# We can substitute `&` for `and` and `|` for `or`.

# In[65]:


True & True


# In[66]:


False | True


# ### Type casting

# We can "recast" variables to change their types.

# In[67]:


s = '3.14159'


# In[68]:


type(s)


# In[69]:


1 + float(s)


# In[70]:


fval = float(s)


# In[71]:


type(fval)


# In[72]:


int(fval)


# Zero is Boolean `False`, and all other values are Boolean `True`.

# In[73]:


bool(0)


# In[74]:


bool(1)


# In[75]:


bool(-1)


# We can recast the string `'5'` to an integer or the integer `5` to a string to prevent the `5 + '5'` error above.

# In[76]:


5 + int('5')


# In[77]:


str(5) + '5'


# ### None

# In Python, `None` is null.
# `None` is like `#N/A` or `=na()` in Excel.

# In[78]:


a = None
a is None


# In[79]:


b = 5
b is not None


# In[80]:


type(None)


# ## Control Flow

# > Python has several built-in keywords for conditional logic, loops, and other standard control flow concepts found in other programming languages.
# 
# If you understand Excel's `if()`, then you understand Python's `if`, `elif`, and `else`.

# ### if, elif, and else

# In[81]:


x = -1


# In[82]:


type(x)


# In[83]:


if x < 0:
    print("It's negative")


# Single quotes and double quotes (`'` and `"`) are equivalent in Python.
# However, in the previous code cell, we use double quotes to differentiate between the enclosing quotes and the apostrophe in `"It's"`.

# Python's `elif` avoids Excel's nested `if()`s.
# `elif` continues an `if` block, and `else` runs if the other conditions are not met.

# In[84]:


x = 10
if x < 0:
    print("It's negative")
elif x == 0:
    print('Equal to zero')
elif 0 < x < 5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')


# We can combine comparisons with `and` and `or`.

# In[85]:


a = 5
b = 7
c = 8
d = 4
if a < b or c > d:
    print('Made it')


# ### for loops

# We use `for` loops to loop over a collection, like a list or tuple.
# The `continue` keyword skips the remainder of the `if` block for that loop iteration.
# 
# The following example assigns values with `+=`, where `a += 5` is an abbreviation for `a = a + 5`.
# There are equivalent abbreviations for subtraction, multiplication, and division (`-=`, `*=`, and `/=`).

# In[86]:


sequence = [1, 2, None, 4, None, 5, 'Alex']
total = 0
for value in sequence:
    if value is None or type(value) is str:
        continue
    total += value # the += operator is equivalent to "total = total + value"


# In[87]:


total


# The `break` keyword exits the loop altogether.

# In[88]:


sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value


# In[89]:


total_until_5


# ### range

# > The range function returns an iterator that yields a sequence of evenly spaced
# integers.
# 
# - With one argument, `range()` creates an iterator from 0 to that number *but excludes that number* (so `range(10)` is an interator with a length of 10 that starts at 0)
# - With two arguments, the first argument is the *inclusive* starting value, and the second argument is the *exclusive* ending value
# - With three arguments, the third argument is the iterator step size

# In[90]:


range(10)


# We can cast a range to a list.

# In[91]:


list(range(10))


# In[92]:


list(range(1, 10))


# In[93]:


list(range(1, 10, 1))


# In[94]:


list(range(0, 20, 2))


# Python intervals are "closed" (inclusive) on the left and "open" (exclusive) on the right.
# The following is an empty list because we cannot count from 5 to 0 by steps of +1.

# In[95]:


list(range(5, 0))


# However, we can count from 5 to 0 in steps of -1.

# In[96]:


list(range(5, 0, -1))


# For loops have the following syntax in many other programming languages.

# In[97]:


seq = [1, 2, 3, 4]
for i in range(len(seq)):
    val = seq[i]


# However, in Python, we can directly loop over the list `seq`.
# The following code cell is equivalent to the previous code cell and more "Pythonic".

# In[98]:


for i in seq:
    val = i


# ### Ternary expressions

# We said above that Python `if` and `else` is cumbersome relative to Excel's `if()`.
# We can complete simple comparisons on one line in Python.

# In[99]:


x = 5
value = 'Non-negative' if x >= 0 else 'Negative'

