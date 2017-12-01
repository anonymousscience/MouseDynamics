import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Statistical functions
# [start, stop)
def mean(array, start, stop):
    avg = 0
    counter = 0
    for i in range(start, stop):
        avg += array[i]
        counter += 1
    if counter == 0:
        return 0
    return avg / counter

# [start, stop)
def stdev(array, start, stop):
    m = mean(array, start, stop)
    n = stop - start
    if n-1 <= 0:
        return 0
    s = 0

    for i in range(start+1, stop):
        s += (array[ i ]-m)*(array[i]-m)
    return math.sqrt(s/(n-1))

# [start, stop)
def max(array, start, stop):
    n = stop - start
    if n-1 <= 0:
        return 0
    maxValue = array[start]
    for i in range(start, stop):
        if array[i] > maxValue:
            maxValue = array[ i ]
    return maxValue

# [start, stop)
def min(array, start, stop):
    n = stop - start
    if n-1 <= 0:
        return 0
    minValue = array[start]
    for i in range(start, stop):
        if array[i] < minValue:
            minValue = array[ i ]
    return minValue

def containsNull ( x ):
    for i in range(0,len(x)):
        if x[i] == 0:
            return True
    return False

