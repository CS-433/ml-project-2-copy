import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Subset
from torch.utils.data import TensorDataset, random_split, SequentialSampler
import numpy as np
import time
import datetime
from tqdm import tqdm
import csv

def train_val_split(dataset, proportion = 0.8):
    '''
    Split the dataset into a validation and a test dataset
    
    Inputs :
        dataset (Pytorch dataset) : dataset that will be split
        proportion (float) : proportion used to split the dataset
        
    Outputs :
        train_ds (Pytorch dataset) : train dataset
        val_ds (Pytorch dataset) : validation dataset
    '''
    full_size = len(dataset)
    train_size = int(proportion * full_size)
    val_size = full_size - train_size

    # Divide the dataset randomly into train and validation sets of size 80%/20% of full dataset
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return train_ds, val_ds

def as_dataloader(dataset, batch_size = 32, random = True):
    '''
    Creates a pytorch dataloader object from a pytorch dataset. Can be created using a sequential or
    random sampler. The sampler will either, shuffle the batches (not the samples in the batches) between epochs,
    Or just feed them sequentially.
    
    Inputs :
        dataset : dataset that be used to create the dataloader object 
                  ( for example, the training or validation dataset)
        batch_size (int) : batch size used to creat the dataloader object
        random (bool) : use the random sampler, otherwise the sequential sampler 
    Outputs : 
        DataLoader object
    '''
  # batch size should be 16 or 32 according to BERT authors
  # random determines id the batches will be selected ransomly or sequentially
    if(random):
        return DataLoader(dataset, sampler = RandomSampler(dataset),batch_size = batch_size)
    else:
        return DataLoader(dataset, sampler = SequentialSampler(dataset),batch_size = batch_size)
    
# Currently not used but could be in the future
#def load_model(device):
#  # Load pretrained BERT model with single linear classification layer on top. 
#  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False)
#
#  # Tell pytorch to run this model on the GPU.
#  model.to(device) #model.cuda()
#  
#  return model
  
def set_seed(seed_val):
    '''
    Function to set the seed
    input:
        seed_val(int) : value used to set the seed
        
    output:
        None
    '''
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# Function to compute the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    '''
    Returnes the flat accuracy between preds and labels
    
    Inputs
        preds (np.ndarray) : array containing the predictions
        labels (np.ndarray) : array containing the correct labels
        
    Outputs 
        (float) : number of correct labels divided by total number of labels
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    
    Input:
        elapsed (float) : elapsed time in seconds 
    Output:
        (str) elapsed time string in hh:mm:ss format
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def make_prediction(model, test_dataloader,device):
    '''
    Take the model, test dataloader and device and uses the model to
    predict the outcome of every batch.
    
    Inputs :
        model (nn.Module)
        test_dataloader (DataLoader)
        device
    Outputs :
        
    
    '''
    prediction = []
    for batch in tqdm(test_dataloader):
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                      token_type_ids=None, # DONT FORGET to fix this line, not requirte for BERT but required for BERT+custom
                                      attention_mask=b_input_mask
                                      ,labels=b_labels)

                logits = outputs.logits 

            logits = logits.detach().cpu()
            prediction.append(logits)
    y_pred = torch.cat(prediction, dim=0)
    y_pred_flat = np.argmax(y_pred, axis=1).flatten()
    y_pred_flat.numpy()
    y_pred_flat[y_pred_flat==0]=-1
    ids = np.arange(len(y_pred_flat))+1
    return y_pred_flat, ids
    

def pred_sanity_checks(y_pred):
    '''
    This function perfroms some simple sanity checks so to check 
    if the predictions are in the correct format
    It will print the number of occurences of -1 , 1, the total number
    of predictions, a sample of the predictions, and what values do the
    prediction logits take.
    
    input:
        y_pred (np.ndarray) : array containing the predictions
        
    output:
        None
    
    '''
    # sanity check
    print((y_pred==-1).sum())
    print((y_pred==1).sum())
    print((y_pred==-1).sum() + (y_pred==1).sum())
    print(y_pred[0:15])
    print('unique values', np.unique(y_pred))

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
               
    This functions has been provided to us by the course in the project 1 helper files
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

''' for loading and saving'''
def save_tokenize(input_ids, attention_masks, labels, PATH_PREPROCESSING,small_dataset=1):
    '''
    Save the input_ids, attention masks, and labels of the tokenized tweets to disk for later use
    
    Inputs :
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the
        labels (tensor)  : tensor containing the labels
        PATH_PREPROCESSING (str) : path in to folder where the files will be saved
        small_dataset (bool) : to indicate wether it's data for the small or large dataset
        
    Outputs :
        None
    '''
    if small_dataset:
        # small dataset
        torch.save(input_ids, PATH_PREPROCESSING +'input_ids_BERT.pkl', _use_new_zipfile_serialization=True)
        torch.save(attention_masks, PATH_PREPROCESSING +'input_ids_BERT.pkl', _use_new_zipfile_serialization=True)
        torch.save(labels, PATH_PREPROCESSING +'input_ids_BERT.pkl', _use_new_zipfile_serialization=True)
    else:
        # full dataset
        torch.save(input_ids, PATH_PREPROCESSING +'input_ids_custombert_full.pkl', _use_new_zipfile_serialization=True)
        torch.save(attention_masks, PATH_PREPROCESSING +'attention_masks_custombert_full.pkl', _use_new_zipfile_serialization=True)
        torch.save(labels, PATH_PREPROCESSING +'labels_custombert_full.pkl', _use_new_zipfile_serialization=True)

def load_tokenize( PATH_PREPROCESSING,small_dataset=1):

    '''
    load the input_ids, attention masks, and labels of the tokenized tweets from disk
    
    Inputs :
        PATH_PREPROCESSING (str) : path in to folder where the files will be saved
        small_dataset (bool) : to indicate wether it's data for the small or large dataset
    
    Outputs :
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the
        labels (tensor)  : tensor containing the labels
    '''
    if small_dataset:
        input_ids = torch.load(PATH_PREPROCESSING +'input_ids_custombert.pkl', map_location=None)
        attention_masks = torch.load(PATH_PREPROCESSING +'attention_masks_custombert.pkl', map_location=None)
        labels = torch.load(PATH_PREPROCESSING +'labels_custombert.pkl', map_location=None)
    else:
        input_ids = torch.load(PATH_PREPROCESSING +'input_ids_custombert_full.pkl', map_location=None)
        attention_masks = torch.load(PATH_PREPROCESSING +'attention_masks_custombert_full.pkl', map_location=None)
        labels = torch.load(PATH_PREPROCESSING +'labels_custombert_full.pkl', map_location=None)
        
    return input_ids, attention_masks, labels