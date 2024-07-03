# Importing the NumPy library with an alias 'np'
import numpy as np

# Creating a NumPy array 'x' with a 2x2 shape
x = np.transpose(np.ones((100,2809)))
y = np.ones((100,))*3

print(x*y)
# Printing a message indicating the original array 'x'
print("Original array:")
print(x)

# Calculating and printing the sum of all elements in the array 'x' using np.sum()
print("Sum of all elements:")
print(np.sum(x))

# Calculating and printing the sum of each column in the array 'x' using np.sum() with axis=0
print("Sum of each column:")
print(np.sum(x, axis=0))

# Calculating and printing the sum of each row in the array 'x' using np.sum() with axis=1
print("Sum of each row:")
print(np.sum(x, axis=1)) 