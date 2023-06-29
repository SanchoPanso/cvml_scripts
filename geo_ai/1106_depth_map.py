import numpy as np
import laspy
import open3d as o3d
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree

data = laspy.open(r'C:\Users\HP\Downloads\pano5\Run_5s2224605_20200609-091004_0005.las')
las = data.read()
print(las.xyz.shape)

point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

X = point_data
tree = KDTree(X, leaf_size=1000)     
ind = tree.query_radius(X[:1], r=0.3)  
print(ind) 
