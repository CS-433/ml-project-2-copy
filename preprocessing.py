from tqdm import tqdm

import pandas as pd
import numpy as np
import json

import nltk
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import webtext
from nltk.probability import FreqDist

tqdm.pandas()


'''Lowercasing'''

def prep_lowercase(df, col = 'Tweet'):
    """
    Lowercase the sentences 
    Inputs:
    - df (DataFrame) : dataframe containing the tweets
    - col (string) : name of the column to lowercase by default to 'Tweet'
    Outputs:    
    - df_new (Serie) : pandas Serie containing the sentence lowercased
    """
    df_new = df['Tweet'].progress_apply(lambda x : x.lower())
    return  df_new

''' Removing contractions'''


#The contraction were obtained on this website that got them from Wikipedia
#https://gist.githubusercontent.com/Sirsirious/c70400176a4532899a483e06d72cf99e/raw/e46fa7620c4f378f5bf39608b45cddad7ff447a4/english_contractions.json


def load_contractions(PATH_PREPROCESSING):
    """
    Loads the contraction list
    Inputs:
    - PATH_PREPROCESSING (string): string containing the path of the folder in which is the contraction list
    Outputs:    
    - contraction_list (dict) : dictionnary containing the contractions as key and their corresponding English word as value
    """
    contraction_list = json.loads(open(PATH_PREPROCESSING + 'english_contractions.json', 'r').read())
    
    #adding two contractions found by scrolling through the tweets
    contraction_list['ur'] = "your"
    contraction_list['ya'] = "you"
    return contraction_list

def remove_contractions(sentence, contraction_list):
    """
    Removes the contractions from a sentence
    Inputs:
    - sentence (string): initial sentence with the contractions
    - contraction_list (dict) : dictionnary containing the contractions as key and their corrsponding English word as value
    Outputs:    
    -" ".join(sent_wo) (string) : string containing the tweet with the contractions replaced
    """
    sent_split = sentence.split(" ")
    sent_wo = [contraction_list[word] if word in contraction_list else word for word in sent_split]
    return " ".join(sent_wo)

def remove_contractions_df(df, contraction_list,  col = 'Tweet'):
    """
    Removes the contractions from the whole dataframe containing the tweets
    Inputs:
    - df (DataFrame): dataframe containing the tweets
    - contraction_list (dict) : dictionnary containing the contractions as key and their corrsponding English word as value
    - col (string) : column containing the Tweet in which to replace the contractions by default to 'Tweet'
    Outputs:    
    - df_new (pandas.Serie) : Serie containing the tweets without the contractions
    """
    
    df_new = df[col].progress_apply(lambda x : remove_contractions(x, contraction_list))
    return df_new

'''Removing slang'''

#The slang dictionnary was obtained on this website
#https://aclanthology.org/P11-1038/


def slang_dict_to_tuple(PATH_PREPROCESSING):
    """
    Transforms a .dict file into a list of tuples
    Inputs:
    - PATH_PREPROCESSING (string): string containing the path to the folder containing the .dict file
    Outputs:    
    - lines (list) : list of tuples containing the slang and their corresponding English word
    """
    
    f = open(PATH_PREPROCESSING + 'slang.dict')
    lines = []
    for line in f:   ## iterates over the lines of the file
        lines.append(line)
    f.close()
    return lines



def slang_tuple_to_dict(lines):
    #https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    """
    From a list of tuples creates a dictionnary
    Inputs:
    - lines (list): list of tuples containing the slang and their corresponding English word
    Outputs:    
    - slang_dict (dict) : dictionnary containing as keys the sland and as values their corresponding English word
    """
    slang_dict = {}
    for i in range(0, len(lines), 2):
        slang_dict[lines[i].replace('\n', '')] = lines[i+1].replace('\n', '').strip()
    return slang_dict

def slang_tuple_to_json(slang_dict,PATH_PREPROCESSING):
    """
    Creates a json file from a dictionnary
    Inputs:
    - slang_dict (dict) : dictionnary containing as keys the sland and as values their corresponding English word
    - PATH_PREPROCESSING (string) : path to the folder where to store the created json
    Outputs:    
    - print('file saved') : prints that it saved the file to PATH_PREPROCESSING +'slang.json'
    """
    with open(PATH_PREPROCESSING + 'slang.json', 'w') as fp:
        json.dump(slang_dict, fp)
    return print('file saved')
  
def remove_slang(sentence, slang_list):
    """
    Removes the slang from a sentence
    Inputs:
    - sentence (string) : initial sentence with the slang
    - slang_dict (dictionnary) : dictionnary containing the slang and their corresponding English word
    Outputs:    
    - " ".join(sent_wo) (string) : string containing the initial sentence with the slang removed
    """
    sent_split = sentence.split(" ")
    sent_wo = [slang_list[word] if word in slang_list else word for word in sent_split]
    return " ".join(sent_wo)

def remove_slang_df(df, slang_list, col = 'Tweet'):
    """
    Removes the slang from the whole dataframe
    Inputs:
    - df (DataFrame) : dataframe containing the Tweets with the slang
    - slang_dict (dictionnary) : dictionnary containing the slang and their corresponding English word
    - col (string) : column containing the Tweet in which to replace the slang by default to 'Tweet'
    Outputs:    
    - df_new (Pandas.Serie) : series containing all the tweets without the slang
    """
    df_new = df[col].progress_apply(lambda x : remove_slang(x, slang_list=slang_list))
    return df_new
    
    
    
'''Remove stopwords'''

#https://stackabuse.com/removing-stop-words-from-strings-in-python/
def load_stopwords():
    """
    Loads the stopwords
    Inputs:
    Outputs:    
    - all_stopwords (list) : list containing all the english stopwords downloaded from nltk
    """
    nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    #appending to the list two specific to our dataset stop words that we saw by scrolling through the dataset
    all_stopwords.append('<user>')
    all_stopwords.append('<url>')
    return all_stopwords

def remove_stopwords(sentence, all_stopwords):
    """
    Removes the stopwords from a sentence
    Inputs:
    - sentence (string) : sentence containing stopwords
    - all_stopwords (list) : list containing the stop words
    Outputs:    
    - no_sw (string) : string containing the sentence without the stopwords
    """
    sent_split = sentence.split(" ")  
    all_stopwords = set(all_stopwords) # set() heremakes it at least 10x faster
    sent_without_sw = [word for word in sent_split if not word in all_stopwords]
    no_sw = " ".join(sent_without_sw)
    return no_sw

def remove_stopwords_df(df, all_stopwords, col = 'Tweet'):
    """
    Removes the stopwords from an entire dataframe
    Inputs:
    - df (DataFrame) : DataFrame containing the tweets with the stopwords
    - all_stopwords (list) : list containing the stop words
    - col (string) : column containing the Tweet in which to remove the stopwords by default to 'Tweet'
    Outputs:    
    - df_new (Pandas.Serie) : serie containing the tweets without the stop-words
    """
    df_new = df[col].progress_apply(lambda x : remove_stopwords(x, all_stopwords=all_stopwords))
    return df_new
    
'''Stemming'''


def stem_sent(sentence, stemmer):
    """
    Stems a sentence
    Inputs:
    - sentence (string) : string containing the tweet not stemmed
    - stemmer (nltk.Stemmer) : stemmer from the nltk library
    Outputs:
    - stemming (string) : sentence stemmed
    """
    sent_split = sentence.split(" ")
    sent = [stemmer.stem(w) for w in sent_split]
    stemming = " ".join(sent)
    return stemming

def stem_sent_df(df, stemmer, col = 'Tweet'):
    """
    Stems the entire dataframe containing the tweets
    Inputs:
    - df (DataFrame) : dataframe containing the tweets to stem
    - stemmer (nltk.Stemmer) : stemmer from the nltk library
    - col (string) : column containing the Tweet to stem by default to 'Tweet'
    Outputs:
    - df_new (Pandas.Serie) : serie containing the tweets stemmed
    """
    df_new = df[col].progress_apply(lambda x : stem_sent(x,stemmer=stemmer))
    return df_new
   

    
'''Removing punctuation'''

def remove_punct_df(df, tokenizer, col = 'Tweet'):
    """
    Removes the punctuation in the entire dataframe
    Inputs:
    - df (DataFrame) : dataframe containing the tweets to stem
    - tokenizer (nltk.RegexpTokenizer) : tokenizer from the nltk library that removes the punctuation
    - col (string) : column containing the Tweet in which to replace the punctation by default to 'Tweet'
    Outputs:
    - df_new (Pandas.Serie) : serie containing the tweets without the punctuation
    """
    return df[col].progress_apply(lambda x :" ".join(tokenizer.tokenize(x)))


'''Removing less frequent words'''


def word_occ_tofile(PATH_PREPROCESSING, df,file_name = 'count_word_pos.txt', col ='Tweet'):
    """
    Writes into a file all the words in order to count them later
    Inputs:
    - PATH_PREPROCESSING (string) : path to the folder where to store the number of times each word appears
    - df (DataFrame) : dataframe containing the tweets
    - file_name (string) : by default to 'count_word_pos.txt', name of the file in which to store the count of the words
    - col (string) : column containing the Tweet in which to replace the contractions by default to 'Tweet'
    Outputs:
    """
    #https://www.geeksforgeeks.org/reading-writing-text-files-python/
    with open(PATH_PREPROCESSING + file_name, "w+") as fo :
        for elem in df[col].tolist():
            fo.write(elem + "\n")

            

def low_occuring_words(PATH_PREPROCESSING, file_name = 'count_word_pos.txt' ):
    """
    Filters the words that are appearing less than 5 times in the whole corpus
    Inputs :
    - PATH_PREPROCESSING (string) : path to the folder containing the file with the frequency of the words
    - file_name (string) : name of the file that has the counts by default to 'count_word_pos.txt'
    Outputs :
    - filter_words (list) : list containing the words that appear 5 or more times in the corpus
    """
    d = dict()
    txt = open(PATH_PREPROCESSING + file_name, "r")
    for line in txt:
        # Remove the leading spaces and newline character
        line = line.strip()
  
        # Split the line into words
        words = line.split(" ")
  
        # Iterate over each word in line
        for word in words:
        # Check if the word is already in dictionary
            if word in d:
                # Increment count of word by 1
                d[word] = d[word] + 1
            else:
                # Add the word to dictionary with count 1
                d[word] = 1
  
    print('accounted' + str(d['accounted']))
    #Takes word only if they appear more than 5 times
    filter_words = [word for word, count in d.items() if count>=5]
    return filter_words

#Function to remove words in sentence

def remove_low_words(sentence, lst):
    """
    Removes the words that are appearing less than 5 times in a sentence
    Inputs :
    - sentence (string) : sentence in which we need to remove the low appearing words
    - lst (list) : list containing the high frequency appearing words
    Outputs :
    - " ".join(sent) (string) : string containing the tweets without the words appearing very few times
    """
    sent_split = sentence.split(" ")
    sent = [word for word in sent_split if word in lst]
    return " ".join(sent)

def remove_low_words_df(df, lst, col = 'Tweet' ):
    """
    Removes the low appearing words in the whole dataframe
    Inputs :
    - df (DataFrame) : dataframe containing the tweets where to remove the low-appearing words
    - lst (list) : list containing the low appearing words
    - col (string) : name of the column in which to remove the words, by default to 'Tweet'
    Outputs :
    - df_new (Pandas.Serie) : serie containing the tweets without the low-appearing words
    """
    lst = set(lst) # set makes it a lot faster
    df_new = df[col].progress_apply(lambda x : remove_low_words(x, lst))
    return df_new
    

def preprocessing(df, contraction_list, stemmer, filter_words, slang_list, all_stopwords, tokenizer):
    '''
    All preprocessing steps combined
    Inputs :
    - df (DataFrame) : dataframe containing the tweets
    - contraction_list (dict) : dictionnary containing the english contractions and their corresponding word
    - stemmer (nltk.Stemmer) : stemmer from nltk library to stem the tweets
    - slang_list (dict) : dictionnary containing the slang and their corresponding English word
    - all_stopwords(list) : list containing the English stopwrods
    - tokenizer (nltk.RegexpTokenizer) : tokenizer from the nltk library that removes the punctuation
    Outputs :
    - df (DataFrame) : DataFrame contraining one column with the tweets pre-processed with every step
    '''
    df['Tweet'] = prep_lowercase(df)
    df['Tweet'] = remove_contractions_df(df, contraction_list)
    df['Tweet'] = remove_slang_df(df, slang_list)
    df['Tweet'] = prep_lowercase(df)
    df['Tweet'] = remove_stopwords_df(df, all_stopwords) 
    df['Tweet'] = stem_sent_df(df,stemmer) 
    df['Tweet'] = remove_punct_df(df, tokenizer)
    df['Tweet'] = remove_low_words_df(df, lst = filter_words, col = 'Tweet' )
    return df
    