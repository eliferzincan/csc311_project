from utils import *
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, d, g):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector : Question difficulties.
    :param beta: Vector : Student's abilities.
    :param d: Vector : Discrimination parameter
    :param g: Scalar : Guessing parameter
    :return: float : The negative log likelihood
    """
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    log_likelihood = 0.

    for i, q_id in enumerate(question_id):
        u = user_id[i]
        x = d[q_id] * (theta[u] - beta[q_id])
        log_likelihood += is_correct[i] * np.log(g + np.exp(x)) + \
                          (1 - is_correct[i]) * np.log(1 - g) - np.log(1 + np.exp(x))
    return -log_likelihood


def update_theta_beta(data, lr, theta, beta, disc, g):
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
    :param disc: Vector : Discrimination parameter
    :param g: Scalar : Guessing parameter
    :return: tuple of vectors
    """
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    # Initialize pre-update params
    b = beta
    t = theta
    d = disc

    for i, q_id in enumerate(question_id):
        u = user_id[i]
        c = is_correct[i]
        x = d[q_id] * (t[u] - b[q_id])
        p = sigmoid(x)
        # compute partials for params
        df_dt = ((c * d[q_id] * np.exp(x))/(g + np.exp(x))) - \
                (d[q_id] * sigmoid(x))
        df_db = ((-c * d[q_id] * np.exp(x))/(g+np.exp(x))) + \
                (d[q_id] * sigmoid(x))
        df_dd = ((c * (t[u] - b[q_id]) * np.exp(x))/(g+np.exp(x))) - \
                ((t[u] - b[q_id]) * p)
        # update params
        t[u] += lr * df_dt
        b[q_id] += lr * df_db
        d[q_id] += lr * df_dd

    return theta, beta, d


def irt(data, val_data, lr, iterations, g):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param g: int: pseudo-guessing parameter
    :return: (theta, beta, val_acc_lst)
    """
    nu = max(data["user_id"]) + 1
    nq = max(data["question_id"]) + 1
    theta = np.ones((nu, 1))
    beta = np.ones((nq, 1))
    d = np.ones((nq, 1))

    val_acc_lst = []
    train_acc_lst = []
    iteration = []
    ll_train = []
    ll_valid = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, d=d, g=g)
        ll_train.append(-neg_lld)
        ll_valid.append(-neg_log_likelihood(val_data, theta, beta, d=d, g=g))

        score = evaluate(data=val_data, theta=theta, beta=beta, d=d, g=g)
        score_train = evaluate(data=data, theta=theta, beta=beta, d=d, g=g)
        val_acc_lst.append(score)
        train_acc_lst.append(score_train)
        iteration.append(i)

        print("iteration: {} \t NLLK: {} \t Score: {}".format(i, neg_lld, score))
        theta, beta, d = update_theta_beta(data, lr, theta, beta, d, g)

    return theta, beta, val_acc_lst, train_acc_lst, iteration, ll_train, \
           ll_valid


def evaluate(data, theta, beta, d, g):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param d: Vector : Discrimination Parameter
    :param g: int : Guessing parameter constant
    :return: float
    """
    pred = []
    for i, q_id in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (d[q_id] * (theta[u] - beta[q_id])).sum()
        p_a = g + (1 - g) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    theta, beta, val_accuracies, train_accuracies, iterations, \
    ll_train, ll_valid = irt(train_data, val_data, 0.03, 25, 0.25)
    print(val_accuracies)
    print(train_accuracies)

    plt.plot(iterations, ll_train, label='Train Accuracy')
    plt.show()
    plt.plot(iterations, ll_valid, label='Validation Accuracy')
    plt.show()


if __name__ == "__main__":
    main()
