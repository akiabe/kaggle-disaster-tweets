import numpy as np
import config
import pandas as pd

from feature_generator import process_tweet, build_freqs, extract_features
from models import sigmoid, gradientDescent

from models import lookup, train_naive_bayes

def predict_tweet(tweet, freqs, theta):
    """
    :param tweet: a string
    :param freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    :param theta: (3,1) vector of weights
    :return y_pred: the probability fo a tweet being positive or negative
    """
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return  y_pred

def test_logistic_regression(test_x, freqs, theta):
    """
    :param test_x: a list of tweets
    :param test_y: (m,1) vector with the corresponding labels for the list of tweets
    :param freqs: a dictionary with the frequency of each pair (or tuple)
    :param theta: weight vector of dimension (3,1)
    :return y_hat: predict values
    """
    # the list fo storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    return y_hat

def naive_bayes_predict(tweet, logprior, loglikelihood):
    """
    :param tweet: a string
    :param logprior: a number
    :param loglikelihood: a dictionary of words mapping to numbers
    :return p: the sum of all the loglikelihood of each word in the tweet
    """
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the loglikelihood of that word to the probability
            p += loglikelihood[word]

    return p

def test_naive_bayes(test_x, logprior, loglikelihood):
    """
    :param test_x: a list of tweets
    :param test_y: the corresponding labels for the list of tweets
    :param logprior: the logprior
    :param loglikelihood: a dictionary with the loglikelihood for each word
    :return accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    # return this properly
    accuracy = 0

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    return y_hats

if __name__ == "__main__":
    # load train set
    df = pd.read_csv(config.TRAINING_FILE)

    # fetch features
    train_x = df.text.values
    # change data type np.ndarray to list
    train_x = train_x.tolist()

    # fetch labels
    train_y = df.target.values
    # reshape np.ndarray (m x 1) matrix
    train_y = np.reshape(train_y, (train_y.shape[0], 1))

    # load test set
    df_test = pd.read_csv(config.TEST_FILE)

    # fetch features
    test_x = df_test.text.values
    # change data type np.ndarray to list
    test_x = test_x.tolist()

    # create frequency dictionary
    freqs = build_freqs(train_x, train_y)

    ### logistic regression ###

    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))

    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

    # make predictions on test set
    lr_y_hat = test_logistic_regression(test_x, freqs, theta)

    # create submission dataframe
    lr_submission = pd.DataFrame(
        np.column_stack((df_test.id.values, lr_y_hat)),
        columns=["id", "target"]
    )

    lr_submission.loc[:,"id"] = lr_submission.loc[:,"id"].astype(int)
    lr_submission.to_csv('../models/lr_submission.csv', index=False)

    ### naive bayes ###

    # calculate the logprior and loglikelihood
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)

    # create predictions for valid set
    nb_y_hat = test_naive_bayes(test_x, logprior, loglikelihood)

    # create submission dataframe
    nb_submission = pd.DataFrame(
        np.column_stack((df_test.id.values, nb_y_hat)),
        columns=["id", "target"]
    )

    nb_submission.loc[:,"id"] = nb_submission.loc[:,"id"].astype(int)
    nb_submission.to_csv('../models/nb_submission.csv', index=False)
