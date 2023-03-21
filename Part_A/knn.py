from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../").toarray()
    val_data = load_valid_csv("../")
    test_data = load_public_test_csv("../")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_lst = [1,6,11,16,21,26]
    user_based_lst = []
    item_based_lst = []
    for k in k_lst:
        user_accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        user_based_lst.append(user_accuracy)
        item_accuracy = knn_impute_by_item(sparse_matrix, val_data, k)
        item_based_lst.append(item_accuracy)
    user_best = report_best_accuracy(user_based_lst)
    k_user = user_based_lst.index(user_best)
    item_best = report_best_accuracy(item_based_lst)
    k_item = item_based_lst.index(item_best)

    print("Highest performance of user based collaborative filtering on the validation data is " + str(user_best) +
          "for k = " + str(k_user))
    print("Highest performance of item based collaborative filtering on the validation data is " + str(item_best) +
          "for k = " + str(k_item))
    test_accuracy_user = knn_impute_by_user(sparse_matrix, test_data, k_user)
    print("Final test accuracy of user based collaborative filtering on test data is " + str(test_accuracy_user))
    test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data, k_item)
    print("Final test accuracy of item based collaborative filtering on test data is " + str(test_accuracy_item))

    plt.plot(k_lst, user_based_lst)
    plt.xlabel("Value of K")
    plt.ylabel("Validation Accuracy for User Based Collaborative Filtering")
    plt.show()

    plt.plot(k_lst, item_based_lst)
    plt.xlabel("Value of K")
    plt.ylabel("Validation Accuracy for Item Based Collaborative Filtering")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def report_best_accuracy(lst: list):
    curr = 0
    for i in lst:
        if i >= curr:
            curr = i
    return curr


if __name__ == "__main__":
    main()
