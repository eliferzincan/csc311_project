from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector : Question difficulties.
    :param beta: Vector : Student's abilities.
    :return: float : The negative log likelihood
    """
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']

    log_likelihood = 0.
    for i, q_id in enumerate(question_id):
        u = user_id[i]
        x = theta[u] - beta[q_id]
        p_a = sigmoid(x)
        log_likelihood += is_correct[i] * np.log(np.exp(p_a)) + \
                          (1 - is_correct[i]) * np.log(1 + np.exp(p_a))

    return -log_likelihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float : Learning Rate for gradient descent
    :param theta: Vector : Student's ability
    :param beta: Vector : Question difficulty
    :param n: Vector
    :return: tuple of vectors
    """
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    # Initialize the original thetas betas that we are trying to update
    b = beta
    t = theta

    # Iterating over each question and updating parameters based on q
    # our update rules are: bj <- bj - a * df_db
    # and tj <- tj - a * df_dt
    # df_db = for j in questions: -cij + sigma(t_i - b_j)
    # df_dt = for i in students: cij - sigma(t_i - b_j)
    for i, q in enumerate(question_id):
        u = user_id[i]
        c = is_correct[i]
        x = t[u] - b[q]
        p = sigmoid(x)
        # compute partials (df_dt,df_db store results from part a))
        df_dt = c - p
        df_db = -c + p
        # perform gradient update
        t[u] += lr * df_dt
        b[q] += lr * df_db

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    nu = max(data["user_id"]) + 1
    nq = max(data["question_id"]) + 1
    theta = np.ones((nu, 1))
    beta = np.ones((nq, 1))

    val_acc_lst = []
    train_acc_lst = []
    iteration = []
    ll_train = []
    ll_valid = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        ll_train.append(-neg_lld)
        ll_valid.append(-neg_log_likelihood(val_data, theta, beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_acc_lst.append(evaluate(data, theta, beta))
        iteration.append(i)
        val_acc_lst.append(score)
        print("iteration: {} \t NLLK: {} \t Score: {}".format(i, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, train_acc_lst, iteration, ll_train, ll_valid


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # print(sparse_matrix.shape)  # 542 students and 1774 questions
    # print(sparse_matrix)

    # train the model lr = 0.01, it = 35
    theta, beta, val_accuracies, train_accuracies, iterations, ll_train, \
    ll_valid = irt(train_data, val_data, 0.008, 19)
    # lr = 0,008 it = 19 has worked best so far: test = 0.7335238498447644
    # 0.7074513124470787
    print(val_accuracies)  # 0.7080158058142817
    print(train_accuracies)  # 0.7325183460344341

    plt.plot(iterations, ll_train, label='Train Accuracy')
    plt.show()
    plt.plot(iterations, ll_valid, label='Validation Accuracy')
    plt.show()


    # Report the training curve showing training and validation log-likelihoods
    # as a function of iteration
    # in irt method

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()