from Part_A.knn import knn_impute_by_item
from utils import *
import numpy as np
import random
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix
from sklearn.impute import KNNImputer


def bootstrap(data):
    n = len(data["question_id"])
    resample = {"question_id": [], "user_id": [], "is_correct": []}
    for i in range(n):
        # choose a random index with replacement
        index = random.randint(0, n - 1)
        # append the corresponding row to the resampled data
        resample["question_id"].append(data["question_id"][index])
        resample["user_id"].append(data["user_id"][index])
        resample["is_correct"].append(data["is_correct"][index])
    # create sparse matrix from resampled data
    sparse_matrix_bootstrapped = csr_matrix((resample["is_correct"], (resample["user_id"], resample["question_id"])),
                                            shape=(542, 1774)).toarray()
    return resample, sparse_matrix_bootstrapped


def bagging_knn_user(train_data, sparse_matrix, valid_data, bag_number, k):
    total = 0
    for i in range(bag_number):
        resample, sparse_matrix_bootstrapped = bootstrap(train_data)
        nbrs = KNNImputer(n_neighbors=k)
        mat = nbrs.fit_transform(sparse_matrix_bootstrapped)
        total += mat
    avg = total / bag_number
    nbrs = KNNImputer(n_neighbors=k)
    avg_fit = nbrs.fit_transform(avg)
    acc = sparse_matrix_evaluate(valid_data, avg_fit)
    return acc


def main():
    sparse_matrix = load_train_sparse("../").toarray()
    train_data = load_train_csv("../")
    val_data = load_valid_csv("../")
    test_data = load_public_test_csv("../")

    accuracy = bagging_knn_user(train_data, sparse_matrix, val_data, 3, 11)
    print("Accuracy after bagging is " + str(accuracy))


if __name__ == "__main__":
    main()
