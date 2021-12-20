import pandas as pd
import torch
def load_tweets(PATH_DATA, small_dataset=1):
    # small dataset
    if small_dataset: 
        train_pos = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_pos.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
        train_neg = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_neg.txt', 
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
    # full dataset
    else: 
        train_pos = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_pos_full.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
        train_neg = pd.read_fwf(PATH_DATA + 'twitter-datasets/train_neg_full.txt',
                                header = None, names = ['Tweet'], colspecs = [(0,280)])
    return train_pos, train_neg


# Following: http://mccormickml.com/2019/07/22/BERT-fine-tuning/#11-using-colab-gpu-for-training
#     using: https://github.com/huggingface/transformers 
def gpu_cpu_setup():
    
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
    Creating a dataset in correct format
    '''
    df_1 = train_pos.copy()
    df_1.rename(columns={"Tweet": "text"},inplace=True)
    df_1['label'] = 1
    df_2 = train_neg.copy()
    df_2.rename(columns={"Tweet": "text"},inplace=True)
    df_2['label'] = 0
    df = pd.concat([df_1, df_2], ignore_index=True, sort=False)
    return df