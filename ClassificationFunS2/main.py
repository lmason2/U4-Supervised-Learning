import operator
import math
import random 
import numpy as np 
import scipy.spatial.distance as distance 

def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist 

# unit tests with pytest start with test_
def test_compute_euclidean_distance():
    # test our euclidean distance against scipy's
    # we need some data
    v1 = np.random.normal(0, 25, 1000)
    v2 = np.random.normal(100, 5, 1000)
    dist = compute_euclidean_distance(v1, v2)
    sp_dist = distance.euclidean(v1, v2)
    assert np.isclose(dist, sp_dist)

def train_test_split(X, y, test_size):
    # TODO: shuffle (randomize) X and y in parallel first
    num_instances = len(X) # 8
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) # ceil(8 * 0.33)
    split_index = num_instances - test_size # 8 - 2 = 6

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None: 
            parallel_list[i], parallel_list[rand_index] = \
                parallel_list[rand_index], parallel_list[i]


def main():
    header = ["att1", "att2"]
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test = [2, 3]

    k = 3 
    # compute the distance from each training instance to [2, 3]
    # append the distance to each training instance
    for i, instance in enumerate(train):
        # append the class label
        instance.append(train_labels[i])
        # append the original row index
        instance.append(i)
        # append the distance to [2, 3]
        dist = compute_euclidean_distance(instance[:2], test)
        instance.append(dist)

    for instance in train:
        print(instance)
    
    # sort train by distance
    train_sorted = sorted(train, key=operator.itemgetter(-1))
    print("after sorting")
    for instance in train_sorted:
        print(instance)
    # grab the top k
    top_k = train_sorted[:k]
    print("Top K Neighbors")
    for instance in top_k:
        print(instance)
    # TODO: call get_column() and get_frequencies() to find the majority vote for class label

    # should remove the extra values at the end of each instance
    for instance in train:
        del instance[-3:]
        print(instance)

    # starting with PA4, we are going to implement machine learning algorithms
    # using similar API as sci kit learn
    # each algorithm is its own class
    # the class has a fit() and a predict() method
    # fit() accepts X, y training data and "fits" the model
    # predict() accepts X testing data and "predict" y (class labels) for the test instances
    # X: feature matrix (AKA list of feature vectors AKA list of instances AKA list of lists AKA table)
    # e.g. like train above
    # y: list of a target y values (AKA class labels)
    # e.g. like train_labels above

    # generating a train and test set from a data set
    # 1. holdout method
    # 2. random subsampling
    # 3. k fold cross validation (and variants)
    # 4. bootstrap method

    # 1. holdout method
    # "hold out" instances from the data set to be used for testing
    # train on the rest
    # e.g. test size 2 instances
    # e.g. test size 2:1 train:test split
    # hold out a proportion of the data for testing (0.33)
    # implemented as train_test_split() in PA4
    X_train, X_test, y_train, y_test = train_test_split(train, train_labels, 0.33)
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("y_train:", y_train)
    print("y_test:", y_test)

    # 2. random subsampling
    # do the holdout method k times
    # (note different k than knn)
    # accuracy is the average accuracy over the k holdout runs

    # 3. k fold cross validation
    # we are more intentional about our train/test partitions (AKA folds)
    # create k folds (partitions) of the data
    # where each instance is in one fold
    # each run, we hold out one fold for testing (train on the rest)
    # each instance is in the test set once
    # for fold in folds:
    #   test on the fold
    #   train on the remaining folds (folds - fold)
    # accuracy is how well the classifier predicts over all the folds 

    # variants
    # LOOCV (leave one out cross validation)
    # k = N (num instances)
    # when you need all the training data you can get 
    # (small dataset)

    # stratified k fold cross validation
    # make sure each fold has roughly the same proportion
    # of class labels as the original dataset
    # first, group by class label
    # for each group, distribute the instances one at a time to a fold

    # 4. bootstrap method
    # random subsampling WITH REPLACEMENT
    # number of instances is N
    # create a training set by sampling N instances with replacement
    # ~63.2% of the instances in the training set
    # ~36.8% of the instances will not be in the training set
    # this is the test set
    # see the notes on github for the math intuition

    # warm up task
    randomize_in_place(train, parallel_list=train_labels)
    print(train)
    print(train_labels)

if __name__ == "__main__":
    main()