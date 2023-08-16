import os
os.chdir('/home/jupyter/ntu/csie-cv/hw2')
from PIL import Image
import numpy as np
import copy


result = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,255,255,255,255,255,255,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,255],
          [0,0,0,0,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,255,255,255,0,0,255,0,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,255,255,255,0,0,255,0,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,255,255,255,255,255,255,255,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,255],
          [0,0,0,0,0,0,0,0,0,255,0,0,255,255,255,0,0,0,0,0,0,255],
          [0,0,255,0,0,255,255,0,255,255,0,0,255,255,255,0,0,0,0,0,0,255],
          [255,255,255,0,0,255,255,0,255,0,0,0,0,255,255,0,0,0,0,0,0,255],
          [0,0,255,0,0,255,255,0,255,0,0,255,255,255,0,0,0,0,0,0,0,255],
          [0,0,255,0,0,255,255,0,255,0,0,0,255,255,0,0,0,0,0,0,0,255],
          [0,0,255,0,0,255,255,0,255,0,0,0,0,255,0,0,0,0,0,0,0,255],
          [255,255,255,255,255,255,255,255,255,255,0,0,0,255,0,0,0,0,0,0,0,255],
          [255,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,255],
          [255,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,255],
          [255,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,255],
          [255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255],
          [255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255],
          [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],]

height, width = 24, 22


result = [[255,0,0,255,0,255],
          [255,255,0,255,0,255],
          [255,0,255,255,0,255],
          [0,0,255,0,255,255],]
height, width = 4, 6


for y in range(height):
    for x in range(width):
          result[y][x] = 1 if result[y][x] == 255 else 0


# to avoid indexing -f in python
def guard_ring():
    for y in range(height):
        result = [0] + result[y] + [0]
    
    result = [0 for _ in range(width+2)] + result +  [0 for _ in range(width+2)]
    

#parent = [[x+y*width for x in range(width)] for y in range(height)]
parent = [i for i in range(width*height)]

def parent_():
    return [[parent[x+y*width] for x in range(width)] for y in range(height)]

def yx(y, x):
    return x+y*width

def union_find(x):
    origin_x =x
    while parent[x] != x:
        x = parent[x]
    
    parent[origin_x] = x
    return x


# do connected components
# build parent for the first time
label = 2
for y in range(height):
    for x in range(width):
        
        left_set, up_set = False, False
        if result[y][x] == 1:
            if x > 0 and result[y][x-1] > 0:
                result[y][x] = union_find(yx(y,x-1))
                left_set = True
            
            if y > 0 and result[y-1][x] > 0:
                if left_set:
                    parent[result[y][x]] = union_find(yx(y-1,x))
                else:
                    result[y][x] = result[y-1][x] 

                up_set = True


            if not left_set and not up_set:
                result[y][x] = label
                label += 1
