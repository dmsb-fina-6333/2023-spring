#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 3 - Built-In Data Structures, Functions, and Files

# ## Introduction

# We must understand Python's core functionality to fully use NumPy and pandas.
# Chapter 3 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/book/) discusses Python's core functionality.
# We will focus on the following:
# 
# 1. Data structures
#     1. tuples
#     1. lists
#     1. dicts (also known as dictionaries)
#     1. *we will ignore sets*
# 1. List comprehensions
# 1. Functions
#     1. Returning multiple values
#     1. Using anonymous functions
# 
# ***Note:*** 
# Indented block quotes are from McKinney unless otherwise indicated.
# The section numbers here differ from McKinney because we will only discuss some topics.

# ## Data Structures and Sequences

# > Python's data structures are simple but powerful. Mastering their use is a critical part
# of becoming a proficient Python programmer.

# ### Tuple

# > A tuple is a fixed-length, immutable sequence of Python objects.
# 
# We cannot change a tuple after we create it because tuples are immutable.
# A tuple is ordered, so we can subset or slice it with a numerical index.
# We will surround tuples with parentheses but the parentheses are not always required.

# In[1]:


tup = (4, 5, 6)


# ***Python is zero-indexed, so zero accesses the first element in `tup`!***

# In[2]:


tup[0]


# In[3]:


tup[1]


# In[4]:


nested_tup = ((4, 5, 6), (7, 8))


# ***Python is zero-indexed!***

# In[5]:


nested_tup[0]


# In[6]:


nested_tup[0][0]


# In[7]:


tup = tuple('string')


# In[8]:


tup


# In[9]:


tup[0]


# In[10]:


tup = tuple(['foo', [1, 2], True])


# In[11]:


tup


# In[12]:


# tup[2] = False # gives an error, because tuples are immutable (unchangeable)


# > If an object inside a tuple is mutable, such as a list, you can modify it in-place.

# In[13]:


tup


# In[14]:


tup[1].append(3)


# In[15]:


tup


# > You can concatenate tuples using the + operator to produce longer tuples:
# 
# Tuples are immutable, but we can combine two tuples into a new tuple.

# In[16]:


(1, 2) + (1, 2)


# In[17]:


(4, None, 'foo') + (6, 0) + ('bar',)


# > Multiplying a tuple by an integer, as with lists, has the effect of concatenating together
# that many copies of the tuple:
# 
# This multiplication behavior is the logical extension of the addition behavior above.
# The output of `tup + tup` should be the same as the output of `2 * tup`.

# In[18]:


('foo', 'bar') * 4


# In[19]:


('foo', 'bar') + ('foo', 'bar') + ('foo', 'bar') + ('foo', 'bar')


# #### Unpacking tuples
# 
# > If you try to assign to a tuple-like expression of variables, Python will attempt to
# unpack the value on the righthand side of the equals sign.

# In[20]:


tup = (4, 5, 6)
a, b, c = tup


# In[21]:


a


# In[22]:


b


# In[23]:


c


# In[24]:


(d, e, f) = (7, 8, 9) # the parentheses are optional but helpful!


# In[25]:


d


# In[26]:


e


# In[27]:


f


# In[28]:


# g, h = 10, 11, 12 # ValueError: too many values to unpack (expected 2)


# We can unpack nested tuples!

# In[29]:


tup = 4, 5, (6, 7)
a, b, (c, d) = tup


# In[30]:


a


# In[31]:


b


# In[32]:


c


# In[33]:


d


# #### Tuple methods

# > Since the size and contents of a tuple cannot be modified, it is very light on instance
# methods. A particularly useful one (also available on lists) is count, which counts the
# number of occurrences of a value.

# In[34]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# ### List

# > In contrast with tuples, lists are variable-length and their contents can be modified in-place. You can define them using square brackets [ ] or using the list type function.

# In[35]:


a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)


# In[36]:


a_list


# In[37]:


b_list


# ***Pyhon is zero-indexed!***

# In[38]:


a_list[0]


# #### Adding and removing elements

# > Elements can be appended to the end of the list with the append method.
# 
# The `.append()` method appends an element to the list *in place* without reassigning the list.

# In[39]:


b_list.append('dwarf')


# In[40]:


b_list


# > Using insert you can insert an element at a specific location in the list.
# The insertion index must be between 0 and the length of the list, inclusive.

# In[41]:


b_list.insert(1, 'red')


# In[42]:


b_list


# In[43]:


b_list.index('red')


# In[44]:


b_list[b_list.index('red')] = 'blue'


# In[45]:


b_list


# > The inverse operation to insert is pop, which removes and returns an element at a
# particular index.

# In[46]:


b_list.pop(2)


# In[47]:


b_list


# Note that `.pop(2)` removes the 2 element.
# If we do not want to remove the 2 element, we should use `[2]` to access an element without removing it.

# > Elements can be removed by value with remove, which locates the first such value and removes it from the list.

# In[48]:


b_list.append('foo')


# In[49]:


b_list


# In[50]:


b_list.remove('foo')


# In[51]:


b_list


# In[52]:


'dwarf' in b_list


# In[53]:


'dwarf' not in b_list


# #### Concatenating and combining lists

# > Similar to tuples, adding two lists together with + concatenates them.

# In[54]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# The `.append()` method adds its argument as the last element in a list.

# In[55]:


xx = [4, None, 'foo']
xx.append([7, 8, (2, 3)])


# In[56]:


xx


# > If you have a list already defined, you can append multiple elements to it using the extend method.

# In[57]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])


# In[58]:


x


# ***Check your output! It will take you time to understand all these methods!***

# #### Sorting

# > You can sort a list in-place (without creating a new object) by calling its sort function.

# In[59]:


a = [7, 2, 5, 1, 3]
a.sort()


# In[60]:


a


# > sort has a few options that will occasionally come in handy. One is the ability to pass a secondary sort key—that is, a function that produces a value to use to sort the objects. For example, we could sort a collection of strings by their lengths.
# 
# Before you write your own solution to a problem, read the docstring (help file) of the built-in function.
# The built-in function may already solve your problem faster with fewer bugs.

# In[61]:


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort()


# In[62]:


b # Python is case sensitive, so "He" sorts before "foxes"


# In[63]:


len(b[0])


# In[64]:


len(b[1])


# In[65]:


b.sort(key=len)


# In[66]:


b


# #### Slicing

# ***Slicing is very important!***
# 
# > You can select sections of most sequence types by using slice notation, which in its basic form consists of start:stop passed to the indexing operator [ ].
# 
# Recall that Python is zero-indexed, so the first element has an index of 0.
# The necessary consequence of zero-indexing is that start:stop is inclusive on the left edge (start) and exclusive on the right edge (stop).

# In[67]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq


# In[68]:


seq[5]


# In[69]:


seq[:5]


# In[70]:


seq[1:5]


# In[71]:


seq[3:5]


# > Either the start or stop can be omitted, in which case they default to the start of the sequence and the end of the sequence, respectively.

# In[72]:


seq[:5]


# In[73]:


seq[3:]


# > Negative indices slice the sequence relative to the end.

# In[74]:


seq


# In[75]:


seq[-1:]


# In[76]:


seq[-4:]


# In[77]:


seq[-4:-1]


# In[78]:


seq[-6:-2]


# > A step can also be used after a second colon to, say, take every other element.

# In[79]:


seq[:]


# In[80]:


seq[::2]


# In[81]:


seq[1::2]


# I remember the trick above as `:2` is "count by 2".

# > A clever use of this is to pass -1, which has the useful effect of reversing a list or tuple.

# In[82]:


seq[::-1]


# We will use slicing (subsetting) all semester, so it is worth a few minutes to understand the examples above.

# ### dict

# > dict is likely the most important built-in Python data structure. A more common
# name for it is hash map or associative array. It is a flexibly sized collection of key-value
# pairs, where key and value are Python objects. One approach for creating one is to use
# curly braces {} and colons to separate keys and values.
# 
# Elements in dictionaries have names, while elements in tuples and lists have numerical indices.
# Dictionaries are handy for passing named arguments and returning named results.

# In[83]:


empty_dict = {}
empty_dict


# A dictionary is a set of key-value pairs.

# In[84]:


d1 = {'a': 'some value', 'b': [1, 2, 3, 4]}


# In[85]:


d1['a']


# In[86]:


d1[7] = 'an integer'


# In[87]:


d1


# We access dictionary values by key names instead of key positions.

# In[88]:


d1['b']


# In[89]:


'b' in d1


# > You can delete values either using the del keyword or the pop method (which simultaneously returns the value and deletes the key).

# In[90]:


d1[5] = 'some value'


# In[91]:


d1['dummy'] = 'another value'


# In[92]:


d1


# In[93]:


del d1[5]


# In[94]:


d1


# In[95]:


ret = d1.pop('dummy')


# In[96]:


ret


# In[97]:


d1


# > The keys and values method give you iterators of the dict’s keys and values, respectively. While the key-value pairs are not in any particular order, these functions output the keys and values in the same order.

# In[98]:


d1.keys()


# In[99]:


d1.values()


# > You can merge one dict into another using the update method.

# In[100]:


d1


# In[101]:


d1.update({'b': 'foo', 'c': 12})


# In[102]:


d1


# ## List, Set, and Dict Comprehensions

# We will focus on list comprehensions.
# 
# > List comprehensions are one of the most-loved Python language features. They allow you to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter in one concise expression. They take the basic form:
# > ```python
# > [expr for val in collection if condition]
# > ```
# > This is equivalent to the following for loop:
# > ```python
# > result = []
# > for val in collection:
# >     if condition:
# >         result.append(expr)
# > ```
# > The filter condition can be omitted, leaving only the expression.
# 
# List comprehensions are very [Pythonic](https://blog.startifact.com/posts/older/what-is-pythonic.html).

# In[103]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']


# We could use a for loop to capitalize the strings in `strings` and keep only strings with lengths greater than two.

# In[104]:


caps = []
for x in strings:
    if len(x) > 2:
        caps.append(x.upper())

caps


# A list comprehension is a more Pythonic solution and replaces four lines of code with one.
# The general format for a list comprehension is `[operation on x for x in list if condition]`

# In[105]:


[x.upper() for x in strings if len(x) > 2]


# Here is another example.
# Write a for-loop and the equivalent list comprehension that squares the integers from 1 to 10.

# In[106]:


squares = []
for i in range(1, 11):
    squares.append(i ** 2)
    
squares


# In[107]:


[i**2 for i in range(1, 11)]


# ## Functions

# > Functions are the primary and most important method of code organization and reuse in Python. As a rule of thumb, if you anticipate needing to repeat the same or very similar code more than once, it may be worth writing a reusable function. Functions can also help make your code more readable by giving a name to a group of Python statements.
# >
# > Functions are declared with the def keyword and returned from with the return keyword:
# > ```python
# > def my_function(x, y, z=1.5):
# >     if z > 1:
# >          return z * (x + y)
# >      else:
# >          return z / (x + y)
# > ```
# > There is no issue with having multiple return statements. If Python reaches the end of a function without encountering a return statement, None is returned automatically.
# >
# > Each function can have positional arguments and keyword arguments. Keyword arguments are most commonly used to specify default values or optional arguments. In the preceding function, x and y are positional arguments while z is a keyword argument. This means that the function can be called in any of these ways:
# > ```python
# >  my_function(5, 6, z=0.7)
# >  my_function(3.14, 7, 3.5)
# >  my_function(10, 20)
# > ```
# > The main restriction on function arguments is that the keyword arguments must follow the positional arguments (if any). You can specify keyword arguments in any order; this frees you from having to remember which order the function arguments were specified in and only what their names are.
# 
# Here is the basic syntax for a function:

# In[108]:


def mult_by_two(x):
    return 2*x


# ### Returning Multiple Values

# We can write Python functions that return multiple objects.
# In reality, the function `f()` below returns one object, a tuple, that we can unpack to multiple objects.

# In[109]:


def f():
    a = 5
    b = 6
    c = 7
    return (a, b, c)


# In[110]:


f()


# If we want to return multiple objects with names or labels, we can return a dictionary.

# In[111]:


def f():
    a = 5
    b = 6
    c = 7
    return {'a' : a, 'b' : b, 'c' : c}


# In[112]:


f()


# In[113]:


f()['a']


# ### Anonymous (Lambda) Functions

# > Python has support for so-called anonymous or lambda functions, which are a way of writing functions consisting of a single statement, the result of which is the return value. They are defined with the lambda keyword, which has no meaning other than "we are declaring an anonymous function."
# 
# > I usually refer to these as lambda functions in the rest of the book. They are especially convenient in data analysis because, as you'll see, there are many cases where data transformation functions will take functions as arguments. It's often less typing (and clearer) to pass a lambda function as opposed to writing a full-out function declaration or even assigning the lambda function to a local variable.
# 
# Lambda functions are very Pythonic and let us to write simple functions on the fly.
# For example, we could use a lambda function to sort `strings` by the number of unique letters.

# In[114]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[115]:


strings.sort()
strings


# In[116]:


strings.sort(key=len)
strings


# In[117]:


strings.sort(key=lambda x: x[-1])
strings


# How can I sort by the *second* letter in each string?

# In[118]:


strings


# In[119]:


strings[2]


# In[120]:


strings[2][1]


# In[121]:


strings.sort(key=lambda x: x[1])
strings

