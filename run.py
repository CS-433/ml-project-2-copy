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
    
    PATH_DATA = './data/'
    PATH_PREPROCESSING = PATH_DATA + 'preprocessing/'

    tqdm.pandas()
    
    device = gpu_cpu_setup()
    
    file_name='BERT/BERT_custom_classifier_smallds_epoch_0'
    path_model = PATH_DATA+'models/'+file_name+'.pkl'

    model = load_model(device, PATH_DATA, path_model, model_name = 'BertWithCustomClassifier')
    
    test_dataloader = load_test_data(PATH_DATA, max_len=40)
    
    y_pred, ids = make_prediction(model, test_dataloader,device)
    
    name = 'run_py_submission'
    create_csv_submission(ids, y_pred, PATH_DATA + 'submissions/output_' + name + '.csv' )



if __name__ == '__main__':
    main()
