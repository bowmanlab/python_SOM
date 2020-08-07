
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:56:23 2020

@author: jeff

based on https://visualstudiomagazine.com/articles/2019/01/01/self-organizing-maps-python.aspx?m=1

"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump, load
import time
# note: if this fails, try >pip uninstall matplotlib
# and then >pip install matplotlib

def closest_node(temp_data, som, m_rows, m_cols):
    som_flat = np.reshape(som, (900,6)) 
    ## som is reshaped again to have 900 rows and 6 columns
    som_dist = np.reshape(np.linalg.norm(som_flat - temp_data, axis=1), (30, 30))
    ## 30 x 30 array of euclidean distances between a data point and all the points on the som
    min_loc = np.where(som_dist == np.amin(som_dist))
    ## the coordinates where the euclidean distance from the som to the data point is at a minimum
    min_dist = np.amin(som_dist)
    ## the minimum euclidean distance
    result = (min_loc[0][0], min_loc[1][0], min_dist)
    ## I do not understand the result line and why is it hardcoded
    
    ## You are here, also return distance, in the expectation that distance
    ## should be steadily decreasing as the map trains.

    return result

def euc_dist(v1, v2):
    return np.linalg.norm(v1 - v2) 

def manhattan_dist(r1, c1, r2, c2):
    return np.abs(r1-r2) + np.abs(c1-c2)

def most_common(lst, n):
    # lst is a list of values 0 . . n
    if len(lst) == 0:
        return -1
    counts = np.zeros(shape=n, dtype=np.int)
    for i in range(len(lst)):
        counts[lst[i]] += 1
    return np.argmax(counts)

def update_map(j):
    if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
        som[i][j] = som[i][j] + curr_rate * (temp_data - som[i][j])

## load data

print("\nLoading Iris data into memory \n")
data_file = "test_AF.training_events.csv"
data_x = np.loadtxt(data_file, delimiter=",", usecols=range(1,7), skiprows = 1, dtype=np.float64)
## data_x is raw data
#data_y = np.loadtxt(data_file, delimiter=",", usecols=[4], dtype=np.int)

# 0. get started
present = 5 # how many times should data be presented to map?
np.random.seed(1)
Rows = 30; Cols = 30
RangeMax = Rows + Cols
LearnMax = 0.5
StepsMax = data_x.shape[1] * present
  
print("Constructing a 30x30 SOM")
data_sample = data_x[np.random.choice(data_x.shape[0], size = Rows * Cols, replace = False),:]
## data_sample - array of 900 elements from 0 to approx. 6065 is generated
data_xr = data_x[np.random.choice(data_x.shape[0], size = data_x.shape[0], replace = False),:]
## data_xr - array of approx 6065 elements from 0 to approx 6065 is generated
som = np.reshape(data_sample, (30, 30, 6))
## data_sample array is reshaped to have 30 rows, 30 columns, with a depth of 6

old_time = time.time()
mean_p_dist = []

for p in range(present): 
    p_dist = []
    for s in range(data_xr.shape[0]):
            
        pct_left = 1.0 - ((s * 1.0) / StepsMax)
        curr_range = (int)(pct_left * RangeMax)
        curr_rate = pct_left * LearnMax
        
        temp_data = data_xr[s]
        (bmu_row, bmu_col, min_dist) = closest_node(temp_data, som, Rows, Cols)
        
        for i in range(Rows):
            for j in range(Cols):
                update_map(j)
                
        if s % (StepsMax/10) == 0:
            print("step = ", str(s * (p + 1)), min_dist)
            
        p_dist.append(min_dist)
    
    mean_p_dist.append(sum(p_dist) / len(p_dist))
    print(mean_p_dist[p])
            
net_time = time.time() - old_time
print(net_time)