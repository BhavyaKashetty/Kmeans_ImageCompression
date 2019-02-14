# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:26:51 2018

@author
Name: Bhavya Kashetty
"""
from skimage import io
import numpy as np
import random 
import scipy.misc
import copy
import sys


if len(sys.argv) < 4:
    print "Usage: Kmeans <input-image> <input-image> <k>";
    sys.exit()

impagePath = sys.argv[1] #r'E:/Koala.jpg'
oupImgPath = sys.argv[2]#r'E:/photo1.jpeg'
image = io.imread(impagePath)
io.imshow(image)
#io.show()
noOfClusters =  int(sys.argv[3])#number of clusters
max_iters = 20 #number of times the k-mean should run
rows = image.shape[0]
cols = image.shape[1]
rgb = image.reshape(image.shape[0]*image.shape[1],3)

#find the initial centroids
def init_centroids(rgb,K):
    return random.sample(list(rgb),K)
 
#run k-means algorithm
def kmeans(rgb,c,K):
    idx = copy.deepcopy(rgb)#np.zeros((np.size(rgb,0),1))
    rgbLen = len(rgb)
    classes = {}
    for iter in xrange(0,max_iters):  
        classes =  [[] for p in range(0,K)]
        for i in xrange(0,rgbLen):
            #find the distance from each point to the cluster point
            distance = [ np.linalg.norm(rgb[i] - c[j]) for j in xrange(0,K)]
            #for each point finding the minimum distance to each cluster
            index = distance.index(min(distance))
            idx[i] = c[index]
            classes[index].append(rgb[i])
        c = [np.average(classes[r],axis=0) for r in xrange(0,K)]    
#    rgb = [c[int(idx[z])] for z in xrange(0,rgbLen)]
#    for z in xrange(0,rgbLen):
#        rgb[z] = c[int(idx[z])]
#    rgb = list(map(lambda x: c[int(idx[x])], rgb))
    return idx


initial_centroids = init_centroids(rgb,noOfClusters)
rgb1 = kmeans(rgb,initial_centroids,noOfClusters)
output_image = np.reshape(rgb1, (rows, cols, 3))
scipy.misc.imsave(oupImgPath, output_image)
image_compressed = io.imread(oupImgPath)
io.imshow(image_compressed)
io.show()
    
