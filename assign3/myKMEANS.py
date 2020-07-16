import numpy as np
import math

def kMeans(X, k):
    #first get the dimensions of X:
    d, N = X.shape #in our test case we know that d = 2 and N = 1000

    #first we need to choose random observations for the initial cluster means
    np.random.seed(0) #reset the seed to generate new numbers compared to before https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
    cluster = 0
    meanSelect = []
    means = np.zeros((d, k))    #creates a d x K array to store the means

    for cluster in range(0, k):#generates a random number used to select cluster means
        r = np.random.randint(low = 0, high = (N + 1))
        meanSelect.clear()
        if (r not in meanSelect): #ensures the same observation isn't used twice
            meanSelect.append(r)
            i = 0                   #loop counter i
            for i in range(0, d):
                means[i, cluster] = np.array([X[i, r]])

    #now we have to assign the observations to the estimated means
    labels = np.zeros(N, dtype = int) #creates an N length 1D array for the labels
    distances = []                      #stores distances between all means for each observation


    for obs in range(0, N):                 #loop through each observation
        distances.clear()                   #clear the distances list
        for m in range(0, k):               #loop through each mean
            distances.append(np.sqrt(np.sum((X[:, obs] - means[:, m])**2, axis = 0)))  #https://stackoverflow.com/questions/20184992/finding-3d-distances-using-an-inbuilt-function-in-python
        labels[obs] = np.argmin(distances) #assign a label based on which mean it was closest to


    oldmeans = []  #create an empty d x k array to store previous iteration's means
    oldlabels = [] #create an empty N length 1D array to store previous iteration's labels

    while(not (np.array_equal(oldmeans, means)) and not (np.array_equal(oldlabels, labels))): #while there is a difference between the new and old means and old and new labels
        
        oldmeans = np.copy(means) #track the previous means with oldmeans
        oldlabels = np.copy(labels)  #track the previous labels with oldlabels

        #calculate the new means for each cluster
        
        for cluster in range(0, k):
            clusterData = X[:, labels == cluster]              #extract all of the data points from one cluster
            clusterMean = np.mean(clusterData, axis = 1)
            for i in range(0, d):
                means[i, cluster] = clusterMean[i]              #put each dimension's mean for each cluster into means array

        for obs in range(0, N):                 #loop through each observation
            distances.clear()                   #clear the distances list
            for m in range(0, k):               #loop through each mean
                distances.append(np.sqrt(np.sum((X[:, obs] - means[:, m])**2, axis = 0)))  #https://stackoverflow.com/questions/20184992/finding-3d-distances-using-an-inbuilt-function-in-python
            labels[obs] = np.argmin(distances) #assign a label based on which mean it was closest to

    return means, labels