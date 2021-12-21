import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

def get_text_label_values(df):
    '''
    Return the text and labels separately from the dataframe containing both
    Input
        df (Dataframe): dataframe containing the tweets and the labels
    outputs
        tweets  (np.ndarray) : containing a string in each cell that is the tweet
        labels (np.ndarray)  : the labels corresponding to each tweet
    '''
    tweets = df.text.values
    labels = df.label.values
    return tweets, labels

''' Tokenizers'''
def tokenize_with_encode_plus(df, max_len=140):
    ''''
    Tokenize tweets using the BERT pretrained tokenizer.
    Uses the encode_plus function for tokenization.
    
    inputs:
        df (DataFrame) : dataframe containing the tweets and labels
        max_len : maximum length in tokens to keep ( so only max_len-2 tokens of the tweet will be kept,
                  because two tokens are used to indicate the beggining and end of the sequence)
    outputs:
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the
        labels (tensor)  : tensor containing the labels
    
    '''
    # take the text and label from dataframe
    tweets, labels = get_text_label_values(df)
    # initialize container variables
    input_ids, attention_masks = [], []
    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    for tweet in tqdm(tweets):
        # compute the tokens with the special tokens and the padding

        # use bert tokenizer to tokenize the text, to add 'CLS' token (101) at the start, 'SEP' token(102) and the end, TOO DOOO CHECK THIS
        # map each word to IDS that bert understands ( like 'dog' to '1231')
        # tweets that have more tokens than 40-2 tokens are truncated ( 2 tokens are taken by CLS and SEP), shorter tweets are padded with 0
        # an attention mask composed of 1's and 0's where there are words
        encoded_tweet =  tokenizer.encode_plus(tweet, add_special_tokens = True, max_length = max_len, padding = 'max_length', 
                                            truncation=True, return_attention_mask = True, return_tensors = 'pt')

        # save the input_ids and attention_masks into list
        input_ids.append(encoded_tweet['input_ids'])
        attention_masks.append(encoded_tweet['attention_mask'])

    # transform all data into torch tensors in correct format
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def tokenize_with_autoencoder(df, max_len=140):
    ''''
    Tokenize tweets using the BERT pretrained tokenizer.
    Uses the only the tokenizer function for tokenization, slighty faster than using encode_plus
    
    inputs:
        df (DataFrame) : dataframe containing the tweets and labels
        max_len : maximum length in tokens to keep ( so only max_len-2 tokens of the tweet will be kept,
                  because two tokens are used to indicate the beggining and end of the sequence)
    outputs:
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the
        labels (tensor)  : tensor containing the labels
    
    '''

    tweets, labels = get_text_label_values(df)
    input_ids, attention_masks = [], []
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for tweet in tqdm(tweets):
        encoded_tweet = tokenizer( tweet , add_special_tokens = True, max_length = max_len, padding = 'max_length', 
                                              truncation=True, return_attention_mask = True, return_tensors = 'pt')

        input_ids.append(encoded_tweet['input_ids'])
        attention_masks.append(encoded_tweet['attention_mask'])

    print('encoding done')
    print('concatenating')
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def tokenize_with_autotokenizer_test(df, max_len=140):
    ''''
    Tokenize tweets using the BERT pretrained tokenizer for the dataset, so this time we don't have labels.
    Uses the only the tokenizer function for tokenization, slighty faster than using encode_plus
    
    inputs:
        df (DataFrame) : dataframe containing the tweets and labels
        max_len : maximum length in tokens to keep ( so only max_len-2 tokens of the tweet will be kept,
                  because two tokens are used to indicate the beggining and end of the sequence)
    outputs:
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the

    '''
    tweets = df.text.values
    input_ids, attention_masks = [], []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    for tweet in tqdm(tweets):

        encoded_tweet =  tokenizer(tweet, add_special_tokens = True, max_length = max_len, padding = 'max_length', 
                                              truncation=True, return_attention_mask = True, return_tensors = 'pt')

        input_ids.append(encoded_tweet['input_ids'])
        attention_masks.append(encoded_tweet['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)


    return input_ids, attention_masks 