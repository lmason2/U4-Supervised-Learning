import operator

import numpy as np
import scipy.spatial.distance as distance # distance.euclidean()

def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

# unit tests with pytest should start with test_
def test_compute_euclidean_distance():
    # test our euclidean distance against scipy
    # need some test data
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

    # get the distance each instance in train is to test
    # we will append this distance to the instance so it "moves" with it
    # we also want the label to move with the instance
    # and the original table index
    for i, instance in enumerate(train):
        # append the label
        instance.append(train_labels[i])
        # append the original row index
        instance.append(i) 
        # append the distance
        dist = compute_euclidean_distance(instance[:2], test)
        instance.append(dist)

    for instance in train:
        print(instance)
    
    # now, we want to sort train by distance
    train_sorted = sorted(train, key=operator.itemgetter(-1))
    k = 3
    top_k = train_sorted[:k]
    print("Top K Neighbors")
    for instance in top_k:
        print(instance)

    # TODO: use get_column() and get_frequencies() to get the majority vote label for the
    # test instance

    # note: should remove the extra values we appended to the end of each train instance
    for instance in train:
        del instance[-3:]
        print(instance)

    # some notes for PA4
    # for PA4 and beyond.... we are going to implement
    # our machine learning models following the API of sci kit learn
    # each algorithm is a class
    # with fit() and predict() methods
    # fit() accepts training data and "fits" the model
    # predict() accepts testing data (used for evaluating the model)
    # data is organized into X and y
    # X: feature matrix (table... AKA a list of instances AKA a list of samples AKA a list of rows
    # AKA a list of feature vectors)
    # e.g. like train above
    # y: list of target (class) values
    # e.g. like train_labels above

    # generating training and testing sets from a dataset
    # 1. holdout method
    # 2. random subsampling
    # 3. k fold cross validation (and variants)
    # 4. bootstrap method

    # 1. holdout method
    # "hold out" some data for testing
    # train on the rest
    # e.g. specify a number of instances
    # hold out 2 instances for testing
    # e.g. specify a proportion to hold out
    # e.g. 2:1 train:test split (0.33 for testing)
    # in PA4 as train_test_split()
    


if __name__ == "__main__":
    main()