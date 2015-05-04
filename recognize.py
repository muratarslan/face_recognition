from sklearn.decomposition import RandomizedPCA
import cv2
import glob
import math
import string
import os.path
import numpy as np


IMAGE_RESOLUTION       = 92 * 112  # Image resolution
NUMBER_OF_EIGENFACES   = 10        # Number of images for person
NUMBER_OF_TRAIN_IMAGES = 400       # Total images in dataset

folders   = glob.glob('dataset/*') # Loading dataset
testFaces = glob.glob('test/*')  # Loading Test Image

# Get name from filename
def getName(filename):
    id = string.split(filename, '/')
    return id[1].replace("s", "")


# Convert images for process
def preprocess(filename):
    imageColor = cv2.imread(filename)
    imageGray  = cv2.cvtColor(imageColor, cv2.cv.CV_RGB2GRAY)
    imageGray  = cv2.equalizeHist(imageGray)
    return imageGray.flat


# Find face function
def find():
    
    # Create and array with flatten images X
    # Array with ID of the person on each image y
    X = np.zeros([NUMBER_OF_TRAIN_IMAGES, IMAGE_RESOLUTION], dtype='int8')
    y = []

    # Populate training array with flatten images from subfolders of dataset and names
    z = 0
    for x, folder in enumerate(folders):
        trainFaces = glob.glob(folder + '/*')
        for i, face in enumerate(trainFaces):
            X[z,:] = preprocess(face)
            y.append(getName(face))
            z = z + 1

    # Component analysis on the images
    pca = RandomizedPCA(n_components=NUMBER_OF_EIGENFACES, whiten=True).fit(X)
    X_pca = pca.transform(X)

    # Create an array with flatten images X
    X = np.zeros([len(testFaces), IMAGE_RESOLUTION], dtype='int8')

    # Populate test array with flatten images from subfolder of "test"
    for i,face in enumerate(testFaces):
        X[i,:] = preprocess(face)

    # Run through test images
    for j, refPca in enumerate(pca.transform(X)):
        distances = []
        # Calculate euclidian distance from test image to each of the known images and save distances
        for i, testPca in enumerate(X_pca):
            dist = math.sqrt(sum([diff**2 for diff in (refPca - testPca)]))
            distances.append((dist, y[i]))
        
        name = min(distances)[1]
        print "This person is: " + str(name)
        
