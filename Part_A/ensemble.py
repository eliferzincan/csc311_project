from Part_A.knn import knn_impute_by_item
from utils import *
import numpy as np
import random
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix
from sklearn.impute import KNNImputer


def bootstrap(train_data):
    n = len(train_data["question_id"])
    resample = {"question_id": [], "user_id": [], "is_correct": []}
    # resample each column with a random integer within the range of that column
    for i in range(n):
        index_question = random.randint(min(train_data["question_id"]), max(train_data["question_id"]))
        index_user = random.randint(min(train_data["user_id"]), max(train_data["user_id"]))
        index_correct = random.randint(min(train_data["is_correct"]), max(train_data["is_correct"]))
        resample["question_id"].append(index_question)
        resample["user_id"].append(index_user)
        resample["is_correct"].append(index_correct)
    # create sparse matrix from resampled data
    sparse_matrix_bootstrapped = csr_matrix((resample["is_correct"], (resample["user_id"], resample["question_id"])),
                                            shape=(542, 1774)).toarray()
    return resample, sparse_matrix_bootstrapped


def main():
    sparse_matrix = load_train_sparse("../").toarray()
    train_data = load_train_csv("../")
    val_data = load_valid_csv("../")
    test_data = load_public_test_csv("../")

    # QUESTION: should i use validation data?? which k value should i use? also helper functions in knn to another file
    data_knn, sparse_matrix_knn = bootstrap(val_data)
    accuracy = knn_impute_by_item(sparse_matrix_knn, data_knn, 4)
    print("Accuracy after bagging is " + str(accuracy))


if __name__ == "__main__":
    main()
