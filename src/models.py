import numpy as np

def sigmoid(z):
    """
    :param z: is the input(can be scaler or an array)
    :return h: the sigmoid of z
    """
    h = 1 / (1 + np.exp(-z))

    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    """
    :param x: matrix of features which is (m,n+1)
    :param y: corresponding labels of the input matrix x, dimensions (m,1)
    :param theta: weight vector of dimension (n+1,1)
    :param alpha: learning rate
    :param num_iters: number of iterations to train the model
    :return J: the final cost
    :return theta: the final weight vector
    """
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = (-1./m) * (np.dot((y.T), np.log(h))) + np.dot((1-y).T, np.log(1-h))

        # update the weight theta
        theta = theta - (alpha/m) * (np.dot(x.T, (h-y)))

    J = float(J)

    return J, theta

def lookup(freqs, word, label):
    """
    :param freqs: a dictionary with frequency of each pair
    :param word: the word to look up
    :param label: the label corresponding to the word
    :return n: the number of times the word with its corresponding label appears
    """
    # freqs.get((word, label), 0)
    n = 0

    pair = (word, label)

    if (pair in freqs):
        n = freqs[pair]

    return n

def train_naive_bayes(freqs, train_x, train_y):
    """
    :param freqs: dictionary from (word, label) to how often the word appears
    :param train_x: a list of tweets
    :param train_y: a list of labels corresponding to the tweets
    :return logprior: the log prior
    :return loglikelihood: the log likelihood of naive bayes equation
    """
    loglikelihood = {}
    logprior = 0

    # calculate the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = V_pos = V_neg = 0
    for pair in freqs.keys():
        # if the label is positive
        if pair[1] > 0:
            # increment the count of unique positive words
            V_pos += 1
            # increment the count of positive words by the count for (word, label) pair
            N_pos += freqs[pair]
        # else, the labels is negative
        else:
            # increment the count of unique negative words
            V_neg += 1
            # increment the count of negative words by the count for (word, label) pair
            N_neg += freqs[pair]

    # calculate the number of documents
    D = len(train_y)

    # calculate the number of positive documents
    D_pos = np.sum(train_y)

    # calculate the number of negative documents
    D_neg = D - D_pos

    # calculate logprior
    logprior = np.log(D_pos) / np.log(D_neg)

    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculate the probability of positive and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood