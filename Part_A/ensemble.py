from Part_A.knn import knn_impute_by_user
from utils import *
import random
from scipy.sparse import csr_matrix
from sklearn.impute import KNNImputer
from item_response import irt, evaluate



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


def bagging_knn_user(train_data, valid_data, k):
    avg_train = 0
    for i in range(3):
        resample, sparse_matrix_bootstrapped = bootstrap(train_data)
        nbrs = KNNImputer(n_neighbors=k)
        mat = nbrs.fit_transform(sparse_matrix_bootstrapped)
        avg_train += mat
        valid_acc = knn_impute_by_user(sparse_matrix_bootstrapped, valid_data, k)
    avg_train = avg_train / 3
    nbrs = KNNImputer(n_neighbors=k)
    avg_fit = nbrs.fit_transform(avg_train)
    train_acc = sparse_matrix_evaluate(valid_data, avg_fit)
    return train_acc


def bagging_irt(train_data, valid_data, lr, iterations):
    val_accuracy = 0
    for i in range(3):
        resample, sparse_matrix_bootstrapped = bootstrap(train_data)
        theta, beta, val_acc_lst, train_acc_lst, iteration, ll_train, ll_valid = irt(resample, valid_data, lr,
                                                                                     iterations)
        val_accuracy += evaluate(valid_data, theta, beta)
    return val_accuracy/3



def main():
    sparse_matrix = load_train_sparse("../").toarray()
    train_data = load_train_csv("../")
    val_data = load_valid_csv("../")
    test_data = load_public_test_csv("../")

    accuracy = bagging_knn_user(train_data, val_data, 11)
    print("KNN: Accuracy after bagging is " + str(accuracy))

    val_accuracy = bagging_irt(train_data, val_data, 0.008, 19)
    print("IRT: Accuracy after bagging is " + str(val_accuracy))

if __name__ == "__main__":
    main()
