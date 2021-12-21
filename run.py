from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

import pandas as pd
import time
import datetime
import random
import numpy as np
import csv

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import RandomSampler, DataLoader, Subset
from torch.utils.data import TensorDataset, random_split, SequentialSampler

from tqdm import tqdm

from preprocessing_bert import *
from helpers_bert import *
from models_bert import *
from train_bert import *
from helpers import *


def main():
    '''
    
    '''
    ### PATHS
    PATH_DATA = './data/'
    PATH_PREPROCESSING = PATH_DATA + 'preprocessing/'
    # replace this with the path where you save the model file, ideally in a subdirectory
    # of where you placed run.py
    path_model = PATH_DATA+'models/BERT/BERT_custom_classifier_smallds_epoch_1.pkl'
    path_test_data = PATH_DATA + 'twitter-datasets/test_data.txt'
    path_submission = PATH_DATA + 'submissions/output_run_py_submission.csv'
    ###
    
    # progress bars
    tqdm.pandas()
    
    # setup the device : gpu if available otherwise cpu 
    device = gpu_cpu_setup()
    
    # load the model with the trained parameters from disk
    model = load_model(device, path_model, model_name = 'BertWithCustomClassifier')
    
    # load the test data, tokenize it and put into a dataloader
    test_dataloader = load_test_data(path_test_data, max_len=40)
    
    # make the prediction on the test data using the loaded model
    y_pred, ids = make_prediction(model, test_dataloader,device)
    
    # create the submission csv file
    create_csv_submission(ids, y_pred, path_submission )


if __name__ == '__main__':
    main()
