from matplotlib import pyplot as plt

from Part_B.irt_partb import update_theta_beta, neg_log_likelihood, irt, irt_evaluate
from utils import *
import numpy as np


def irt_bayesian(data, val_data, lr, iterations, g, prior_alpha, prior_beta):
    """ Train IRT model using Bayesian estimation.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float : Learning Rate for gradient descent
    :param iterations: int : Number of iterations for gradient descent
    :param g: float : Pseudo-guessing parameter
    :param prior_alpha: float : Shape parameter of prior distribution for d (discrimination)
    :param prior_beta: float : Scale parameter of prior distribution for d (discrimination)
    :return: tuple of vectors : (theta, beta, d, val_acc_lst)
    """
    nu = max(data["user_id"]) + 1
    nq = max(data["question_id"]) + 1
    iteration = []

    # Initialize theta and beta with part b (parameters added) irt model
    theta, beta, val_acc_lst, train_acc_lst, iterations_other, ll_train, ll_valid = irt(data, val_data, lr, iterations, g)

    # Initialize discrimination with prior distribution
    np.random.seed(66)
    d = np.random.gamma(prior_alpha, prior_beta, size=(nq, 1))
    val_acc_lst_b = []
    train_acc_lst_b = []
    for i in range(iterations):
        # alpha_posterior = prior_alpha + np.sum(theta - beta, axis=0).reshape((nq,))
        # beta_posterior = prior_beta + np.sum(d * np.power(theta - beta, 2), axis=0).reshape((nq,))

        alpha_posterior = prior_alpha + np.sum(theta - beta.reshape((1, -1)), axis=0)
        beta_posterior = prior_beta + np.sum(d.T * np.power(theta - beta.reshape((1, -1)), 2), axis=0)
        if np.any(beta_posterior <= 0):
            beta_posterior = np.clip(beta_posterior, a_min=0.001, a_max=None)
        if np.any(alpha_posterior <= 0):
            alpha_posterior = np.clip(alpha_posterior, a_min=0.001, a_max=None)
        if alpha_posterior.shape != beta_posterior.shape:
            raise ValueError("alpha_posterior and beta_posterior must have the same shape")
        d = np.random.gamma(alpha_posterior, 1 / beta_posterior)

        # Update theta and beta using gradient descent with updated d
        theta, beta, _ = update_theta_beta(data, lr, theta, beta, d, g)

        # Compute and store metrics
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, d=d, g=g)
        ll_train.append(-neg_lld)
        ll_valid.append(-neg_log_likelihood(val_data, theta, beta, d=d, g=g))
        score = irt_evaluate(val_data, theta, beta, d, g)
        score_train = irt_evaluate(data, theta, beta, d, g)
        val_acc_lst_b.append(score)
        train_acc_lst_b.append(score_train)
        iteration.append(i)

        print("iteration: {} \t Score: {}".format(i, score))

    return theta, beta, val_acc_lst, train_acc_lst, iteration, ll_train, \
           ll_valid


def main():
    train_data = load_train_csv("../")
    val_data = load_valid_csv("../")
    test_data = load_public_test_csv("../")

    theta, beta, test_accuracy, train_accuracies, iteration, ll_train, ll_valid = irt_bayesian(train_data, test_data,
                                                                                                0.008,
                                                                                                50, 0.25, prior_alpha=5,
                                                                                                prior_beta=5)
    # best accuracy obtained: train 0.7435788879480666 val 0.7082980524978831 iter=50 priors=5 lr=0.08

    print(test_accuracy)
    print(train_accuracies)

    plt.plot(list(iteration), train_accuracies, label='Train Accuracy')
    plt.show()
    plt.plot(list(iteration), test_accuracy, label='Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
