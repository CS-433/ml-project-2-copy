import numpy as np
import os 
import nltk
import pandas as pd
import gensim.downloader as api
"""
GloVe embeddings
"""

from gensim.models import KeyedVectors
from gensim.downloader import base_dir



tokenizer = nltk.RegexpTokenizer(r"\w+")

def load_glove_model(path_pretrained_embeddings = 'data/twitter-datasets/glove-twitter-25.gz'):
    ''' 
    This loads the 25 dim embeddings models
    Inputs :
    Outputs :
    - model (KeyedVectors) : contains the 25 dim embeddings model
    '''
    path = os.path.join('data/twitter-datasets/glove-twitter-25.gz')
    model = KeyedVectors.load_word2vec_format(path)
    return model


def avg_glove(tweet, model1):
    '''
    Creates the glove vector for each tweet
    Inputs :
    - tweet (string) : contains the string to embed
    - model1 (KeyedVectors) : contains the GloVe model
    Outpus :
    - np.hstack((vec, oov)) (ndarray) : vector corresponding to the tweet embedded
    '''
    #initialize vector :
    vec_init = np.zeros(25)       
    vec = vec_init
    tweet_tok = tokenizer.tokenize(tweet)
    N = len(tweet_tok)
    oov = 0 # out of vocabulary words counter
    for word in tweet_tok :
        try:
            vec += model1[word]
        except: # sometimes the word isn't in the pretrained vocab so use zero vector
            vec += vec_init
            oov +=1
    vec = vec/N
    return np.hstack((vec,oov))

def df_to_GloVe(train_pos, model1):
    '''
    Creates the glove vector for each tweet in the whole dataframe
    Inputs :
    - train_pos (dataframe) : dataframe that contains all the tweets to embed
    - model1 (KeyedVectors) : contains the GloVe model
    Outpus :
    - train_pos_embeddings (dataframe) : dataframe with the tweets embedded
    '''
    df = train_pos.copy()
    df['embeddings'] = train_pos['Tweet'].progress_apply(lambda x : avg_glove(x, model1))
    train_pos_embeddings = pd.DataFrame(df['embeddings'].to_list())
    return train_pos_embeddings

def clean_cols(df):
    '''
    Removes the last column that was used to count the number of words that weren't in the vocabulary and replace the nans and inf     by 0
    Inputs :
    - df (dataframe) : dataframe containing the tweets embedded to clean
    Outputs:
    '''
    # removing the last column ( was used just to test for oov words)
    df.drop(columns=df.columns[-1],inplace=True)

    # replace inf and nan values with 0 ( there were some )
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

"""
TF-IDF embeddings
"""

from sklearn.feature_extraction.text import TfidfVectorizer



def tf_idf_embedding(train_pos, train_neg, max_features=10000, ngram_range=(1,1)):
    '''
    Create the tf_idf_embeddings
    Inputs :
    train_pos (DataFrame) : dataframe containing the positive tweets
    train_neg (DataFrame) : dataframe containing the negative tweets
    max_features (int) : maximum number of features for TF-IDF by default to 10'000
    ngram_range (tuple) : to tell the number of n_grams by default to (1,1)
                            ngram_range = (1,1) only use unigrams
                            ngram_range = (1,3) use unigrams to tri-grams
    Outputs :
    vectors (sparse matrix) : sparse matrix containing the embeddings of the data
    vectorizer (TfidfVectorizer) : vectorizer used for those embeddings
    '''
    df_posneg = train_pos.append(train_neg)
    # take only the 10'000 most common words otherwise too many dimensions
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range) 
    vectors = vectorizer.fit_transform(df_posneg['Tweet']) # creates a sparse matrix
    vectors.shape
    return vectors, vectorizer


def add_label_tfidf(vectors):
    """
    Creates the vectors of label from the vectorized tweets 
    Inputs :
    - vectors(sparse matrix) : contains the data points, the first half being the positive tweets, the second half the negative         ones
    Outputs :
    - x (sparse matrix) : contains the vectorized tweets
    - y (ndarray) : contains the labels of the tweets, 1 for the positive ones 0 for the negative ones
    """
    x = vectors
    n = round(x.shape[0]/2)
    y = np.hstack((np.ones(n),np.zeros(n)))
    return x, y