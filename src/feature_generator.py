import re
import string
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#nltk.download('stopwords')

def process_tweet(tweet):
    """
    :param tweets: a strings containing a tweet
    :return freqs: a list of word containing the processed tweet
    """
    # remove url
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)

    # remove retweet text "RT"
    tweet = re.sub(r'^RT+', '', tweet)

    # remove hashtags
    tweet = re.sub(r'#', '', tweet)

    # instantiate tokenizer class
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )

    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(tweet)

    # import the english stop words list from NLTK
    stopwords_english = stopwords.words('english')

    # instantiate stemming class
    stemmer = PorterStemmer()

    # create empty list to store the clean tweets
    tweets_clean = []

    # remove stop words and punctuations
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """
    :param tweets: a list of tweets
    :param ys:  an (m x 1) array with the sentiment label of each tweet
    :return freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    """
    # create np array to list
    yslist = np.squeeze(ys).tolist()

    # creat empty dictionary to store word frequencies
    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def extract_features(tweet, freqs):
    """
    :param tweet: a list of words for one tweet
    :param freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    :return x: a feature vector of dimension (1,3)
    """
    # process tweet tokenizes, stems, and remove stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a (1 x 3) vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0,0] = 1

    for word in word_l:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1), 0)

        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0), 0)

    assert (x.shape == (1, 3))

    return x

