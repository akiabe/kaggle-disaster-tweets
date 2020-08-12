import argparse

import numpy as np
import pandas as pd

import config
from feature_generator import process_tweet, build_freqs, extract_features
from models import sigmoid, gradientDescent
from predict import predict_tweet, test_logistic_regression

def run(fold):
    # load train set with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # fetch train set that kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # fetch valid set that kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # fetch train features
    train_x = df_train.text.values
    # change data type np.ndarray to list
    train_x = train_x.tolist()

    # fetch train labels
    train_y = df_train.target.values
    # reshape np.ndarray (m x 1) matrix
    train_y = np.reshape(train_y, (train_y.shape[0], 1))

    # similarity, for valid set
    valid_x = df_valid.text.values
    valid_x = valid_x.tolist()

    valid_y = df_valid.target.values
    valid_y = np.reshape(valid_y, (valid_y.shape[0], 1))

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

    # create predictions for valid set
    preds = test_logistic_regression(valid_x, freqs, theta)

    # calculate and print accuracy
    # (# of tweets classified correctly) / (total # of tweets)
    accuracy = np.sum(np.asarray(preds) == np.squeeze(valid_y)) / valid_y.shape[0]
    print(f"Fold={fold}, Accuracy = {accuracy}")

if __name__ == "__main__":
    # instantiate ArgumentParser of argparse
    parser = argparse.ArgumentParser()

    # add fold argument and data type
    parser.add_argument(
        "--fold",
        type=int
    )

    # read the arguments from the command line
    args = parser.parse_args()

    # run the specified fold
    run(
        fold=args.fold
    )