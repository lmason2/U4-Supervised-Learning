import operator
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


if __name__ == "__main__":
    main()