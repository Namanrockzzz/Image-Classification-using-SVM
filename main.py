# Importing Libraries
import numpy as np
import os
from pathlib import Path
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Define SVM class
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
        
    def hingeloss(self, W, b, X, Y):
        loss = 0.0
        loss+=0.5*np.dot(W,W.T)
        m = X.shape[0]
        for i in range(m):
            ti = Y[i]*(np.dot(W, X[i].T)+b)
            loss+=self.C*max(0,(1-ti))
        return loss[0][0]
    
    def fit(self, X, Y, batch_size=100, learning_rate=0.001, maxItr = 300):
        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]
        n = learning_rate
        c = self.C
        # Init the model parameters
        W = np.zeros((1, no_of_features))
        bias = 0
        
        # print(self.hingeloss(W,bias, X, Y))
        
        #start training from here
        # Weight and Bias update rule
        losses = []
        for i in range(maxItr):
            # Training loop
            l = self.hingeloss(W,bias,X,Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids) #shuffled indices
            
            #Batch Gradient Descent using Random shuffeling
            for batch_start in range(0, no_of_samples, batch_size):
                    #Assme 0 gradient for the batch
                    gradw = 0
                    gradb = 0
                    
                    #Iterate over all examples in the mini batch
                    for j in range(batch_start, batch_start+batch_size):
                        if j<no_of_samples:
                            i = ids[j] #shufled indices
                            ti = Y[i]*(np.dot(W,X[i].T)+bias)
                            
                            if ti>1:
                                gradw += 0
                                gradb += 0
                            else:
                                gradw += c*Y[i]*X[i]
                                gradb += c*Y[i]
                    # Gradient for the batch is ready! Update W,B
                    W = W - n*W + n*gradw
                    bias = bias + n*gradb
                    
        self.W = W
        self.b = bias
        return W, bias, losses

# Function to map image_data to is class
def classWiseData(x,y):
    data = {}
    
    for i in range(CLASSES):
        data[i] = []
        
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
    
    for k in data.keys():
        data[k] = np.array(data[k])
    
    return data

# Function for one vs one method of SVM multiclass classifictaion
def getDataPairForSVM(d1, d2):
    """Combines data of two classes into a single matrix"""
    
    l1,l2 = d1.shape[0], d2.shape[0]
    
    samples = l1+l2
    features = d1.shape[1]
    
    data_pair = np.zeros((samples, features))
    data_labels = np.zeros((samples,))
    
    data_pair[:l1,:] = d1
    data_pair[l1:,:] = d2

    data_labels[:l1] = -1
    data_labels[l1:] = 1
    
    return data_pair, data_labels

# Function to train nC2 SVMs
def trainSVMs(x,y):
    
    svm_classifiers = {}
    for i in range(CLASSES):
        svm_classifiers[i] = {}
        for j in range(i+1, CLASSES):
            xpair, ypair = getDataPairForSVM(data[i],data[j])
            wts,b,loss = mySVM.fit(xpair, ypair, learning_rate=0.00001, maxItr=1000)
            svm_classifiers[i][j] = (wts,b)
            
            
            #plt.plot(loss)
            #plt.show()
            
    return svm_classifiers

# Compare data only between a pair
def binaryPredict(x,w,b):
    z = np.dot(x,w.T)+b
    if z>=0:
        return 1
    else:
        return -1

# Predict Function
def predict(x):
    
    count = np.zeros((CLASSES,))
    
    for i in range(CLASSES):
        for j in range(i+1, CLASSES):
            w,b = svm_classifiers[i][j]
            # Take a majority prediction form each of the classifiers
            z = binaryPredict(x,w,b)
            if z==1:
                count[j] += 1
            else:
                count[i] += 1
        
    final_prediction = np.argmax(count)
    # print(count)
    return final_prediction

# Calculate accuracy
def accuracy(x,y):
    
    count=0
    for i in range(x.shape[0]):
        prediction = predict(x[i])
        if prediction==y[i]:
            count+=1
    return count/x.shape[0]
    

# Dataset Preparation
p = Path("Dataset/images/")
# print(type(p))

dirs = p.glob("*")

labels_dict = {"cat":0, "dog":1, "horse":2, "human":3}
image_data = []
labels = []

for folder_dir in dirs:
    # print(folder_name)
    label = str(folder_dir).split("\\")[-1][:-1]
    # print(label)
    
    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path, target_size=(32,32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])

# convert this into numpy array
image_data = np.array(image_data, dtype="float32")/255.0
labels = np.array(labels)

# Randomly shuffle the entire data
import random
combined = list(zip(image_data, labels))
random.shuffle(combined)

#Unzip
image_data[:], labels[:] = zip(*combined)
        
# Reshape the data
M = image_data.shape[0]
image_data = image_data.reshape((M,-1))

# Number of classes
CLASSES = len(np.unique(labels))

# Creating dictionary mapping class to image_data
data = classWiseData(image_data, labels)

# Train the model
mySVM = SVM()
xp, yp = getDataPairForSVM(data[0], data[1])
w,b,loss = mySVM.fit(xp,yp, learning_rate=0.00001, maxItr=1000)
#print(loss)
#plt.plot(loss)
#plt.show()

# Find nC2 svm classifiers
svm_classifiers= trainSVMs(image_data, labels)

# Accuracy
print(accuracy(image_data, labels))

