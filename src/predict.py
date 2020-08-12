import numpy as np
import config
import pandas as pd

from feature_generator import process_tweet, build_freqs, extract_features
from models import sigmoid, gradientDescent

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

    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))

    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

    # make predictions on test set
    y_hat = test_logistic_regression(test_x, freqs, theta)

    #
    submission = pd.DataFrame(
        np.column_stack((df_test.id.values, y_hat)),
        columns=["id", "target"]
    )

    submission.loc[:,"id"] = submission.loc[:,"id"].astype(int)
    submission.to_csv("../models/lr_submission.csv", index=False)
