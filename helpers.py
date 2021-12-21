import pandas as pd
import torch

def load_tweets(PATH_DATA, small_dataset=1):
    '''
    This function loads the training tweets
    
    inputs:
        PATH_DATA (str) : path to folder with train_pos.txt etc
        small_dataset (bool) : use the small or the big dataset
    outputs:
        train_pos (DataFrame) : dataframe containing the positive tweets
        train_neg (DataFrame) : dataframe containing the negative tweets
    '''
    
    if small_dataset: 
        train_pos = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_pos.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
        train_neg = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_neg.txt', 
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
    # otherwise load the full dataset
    else: 
        train_pos = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_pos_full.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
        train_neg = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_neg_full.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
    return train_pos, train_neg



def gpu_cpu_setup():
    '''
    Function to setup the gpu device in pytorch if it is available
    
    outputs:
        device (str) : device names ( either 'cpu' or 'gpu' )
       
    Based on:
    http://mccormickml.com/2019/07/22/BERT-fine-tuning/#11-using-colab-gpu-for-training
    https://github.com/huggingface/transformers 
    '''
    if torch.cuda.is_available():    
        # If there's a GPU available, tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    return device

def create_input_df(train_pos,train_neg):
    '''
    Takes the two dataframes containing positive and negative tweets and 
    returns one dataframe with all the tweets and their corresponding labels
    
    Inputs:
        train_pos (DataFrame) : dataframe containing the positive tweets
        train_neg (DataFrame) : dataframe containing the negative tweets
    Outputs :
        df (DataFrame) : dataframe containing the positive and negative 
                        tweets in the column "Tweet" and the labels the 
                        column 'label'
    '''
    # label the postive tweets
    df_1 = train_pos.copy()
    df_1.rename(columns={"Tweet": "text"},inplace=True)
    df_1['label'] = 1
    
    # label the negative tweets
    df_2 = train_neg.copy()
    df_2.rename(columns={"Tweet": "text"},inplace=True)
    df_2['label'] = 0
    
    # combine both dataframes with tweets and labels
    df = pd.concat([df_1, df_2], ignore_index=True, sort=False)
    return df