# here we load the prepared data
# and build the kmeans algorithm
import numpy as np
import pandas as pd
import os


# load the trainable file
X = np.loadtxt(os.path.join("trainable", "trainit.txt"))

# load the data set
def load_data(name):
    return np.loadtxt(name)

# calculate the eucaldian distance
def dist_method(a,b):
    return np.linalg.norm(a-b)

# the alogorithim hyperparametersa preferably be in text format
# the dat
def kmeans(k,data, epsilion=0):
    # list to store the past centroids
    hist_centroids = []
    
    # set the data set
    # data = load_data(data)
    # get the shape of the data
    n_instances, n_cols = data.shape
    # define k centroids
    prototypes = data[np.random.randint(0, n_instances - 1, size=k)]
    # append the centroids
    hist_centroids.append(prototypes)
    # store the centroids at every iteration
    prototype_old = np.zeros((prototypes.shape))
    # to store clusters
    clusters = np.zeros((n_instances, 1))
    norm = dist_method(prototypes, prototype_old)
    iters = 0
    while norm > epsilion:
        iters +=1
        # calc the norm
        norm = dist_method(prototypes, prototype_old)
        # for each instance in the data set
        for ind_instance, instance in enumerate(data):
            # define the distance
            dist_vec = np.zeros((k,1))
            # for each centorids
            for ind_prototype, prototype in enumerate(prototypes):
                # compuite the dist between x and th ecentroid
                dist_vec[ind_prototype] = dist_method(prototype, instance)
            # find the smallest distance, assign that distacne to a cluster
            clusters[ind_instance, 0] =  np.argmin(dist_vec)

        # list of temporory centroids
        tmp_prototypes = np.zeros((k, n_cols))

        # assign the points to a cluster
        for ind in range(len(prototypes)):
            # get all the points assign to a cluster
            instance_close = [i for i in range(len(clusters)) if clusters[i] == ind]
            # calculate the mean for those points
            prototype = np.mean(data[instance_close], axis=0)
            # add our new centroid to our temporory list
            tmp_prototypes[ind, :] = prototype

        #set the new list ot the old onw
        prototypes = tmp_prototypes
        # store the temporory centorids
        hist_centroids.append(tmp_prototypes)

        

    return prototypes, hist_centroids, clusters


# train the model
print("Start training")
prototypes, hist_centroids, clusters = kmeans(k=2, data=X)

print(clusters)
print("Finished training")

# to predict cluster.
# feed the categorical variable as 0 or 1 as they might be one-hot-vector encoded.
# check the shape of the data to know how many features you're supposed 
# to enter in a list format.
def predict(point):
    prototypes, _, _ = kmeans(k)
    for ind in range(len(prototypes)):
        dist[ind] = eucladian(point, prototype[ind])
    return np.argmin(dist)
