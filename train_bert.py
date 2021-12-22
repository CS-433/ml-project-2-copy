import random
import numpy as np
import torch
import time
from tqdm import tqdm

from helpers_bert import *
import time
import datetime
from models_bert import *
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

def validation(model, val_dataloader, device):
    '''
    Returns validation accuracy and loss 
    
    inputs:
        model (nn.module) : the model to be validated (BERTforsequencclassification or BertWithCustomClassifier)
        val_dataloader (Pytorch Dataloader or bool) : validation dataloader, can be set to None if only training
                                                      desired
        device (str) : 'cpu' or 'gpu' to speed up training
    
    outputs:
        total_eval_accuracy (float) : validation accuracy
        total_eval_loss (float) : validation loss
    '''
    total_eval_accuracy = 0
    total_eval_loss = 0
    # evaluation metrics by batch for better performance
    for batch in tqdm(val_dataloader):
       
        # moving batch to device for faster computation(if gpu)
        batch_input_ids = batch[0].to(device)
        batch_input_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)

        # validation so gradients don't need to be calculated
        with torch.no_grad():        

            # Forward pass, calculate loss and predictions
            outputs = model(batch_input_ids,
                            token_type_ids=None,
                            attention_mask=batch_input_mask,
                            labels=batch_labels)
            
            loss = outputs.loss # Cross entropy loss ( for both BERT and BERT + custom classifier )
            logits = outputs.logits # prediction logits (still not discrete labels)

        # loss of one epoch = sum of batch losses
        total_eval_loss += loss.item()

        # move logits and labels to cpu so we can use numpy
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()

        # accumlate accuracy over batches, so that later it is divided by num_batches to get average accuracy
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    return total_eval_accuracy, total_eval_loss

def train_bert_class_with_params(train_dataloader, val_dataloader, model, 
                                 optimizer, scheduler, epochs, random_seed, device,
                                 PATH_DATA,
                                 save_N_steps=False, save_epoch=False,
                                 save_path = './data/models/BERT/model',step_print=100,
                                 validate = True,
                                 freezing = False,
                                 freez_steps = 100,
                                 frozen_epochs = 1):
    ''' 
    Train function used for the two models : BERTforsequencclassification and our
    version with a custom classifier (BertWithCustomClassifier). This train function
    can freez part of the bert model, save the model every N steps, at the end of every epoch, and
    can perform validation or not.
    
    This training code is based on the `run_glue.py` script here:
    https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    
    Inputs: 
        train_dataloader(Pytorch Dataloader) : data loader for the training data
        val_dataloader (Pytorch Dataloader or bool) : validation dataloader, can be set to None if only training
                                                      desired
        model (nn.module) : the model to be trained (BERTforsequencclassification or BertWithCustomClassifier)
        optimizer () : optimizer from transformers library, such as AdamW
        scheduler () : hugging face scheduler such as the one obtained from 'get_linear_schedule_with_warmup'
                       https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
                       (Can be set to None if validation is set to None)
        epochs (int) : number of epoche
        random_seed (int) : seed so that the training is reproducile
        device (str) : 'cpu' or 'gpu' to speed up training
        PATH_DATA (str) : main data path used to save the model
        save_N_steps (int) : save the model avec N steps
            save_N_steps = False : don't save model every N steps
            save_N_steps = N (type = int) : save model every N steps
        save_epoch (bool) : save the model to disk at the end of every epoch
        txt_header (str) : str that is appened to the model file name
        step_print (int) : print time each N steps ( N=0 disables it)
        validate (bool) : run validation or not
        freezing (bool) : freez BERT layers for freez_steps steps and frozen_epochs epochs ( classifier remains unfrozen)
                          --> ONLY USE FREEZING IF MODEL IS BertWithCustomClassifier, BERTforsequencclassification doesn't have
                          the option
        freez_steps (int) : number of steps during which the BERT layers are frozen
        frozen_epochs (bool) : number of epochs during which the BERT layers are frozen
    
    Outputs:
        training_stats (list) : list of dicts that contain train/validatin loss/accuracy and train/val time for every epoch 
    '''
    # seeds so that experiment is reproductible
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # store training statistics : train/validation loss/accuracy time
    training_stats = []

    # track total train time
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        # beggining of each epoch timer
        t0 = time.time()

        # reset epoch loss to 0
        total_train_loss = 0

        # dropout and batch norm behave differently in train and eval, set to train mode
        model.train()

        # to track train loss and accuracy
        total_train_accuracy = 0
        total_train_loss = 0

        # iterate of training data by batch for performance reasons
        for step, batch in enumerate(tqdm(train_dataloader)):

            if freezing:
                # freeze bert parameters for first 100 steps of first epoch
                if ((epoch_i<frozen_epochs)&(step<freez_steps)):
                    model.freeze_bert(freeze=True)
                    # print('BERT params frozen')
                else:
                    # print('BERT params NOT frozen')
                    model.freeze_bert(freeze=False)

            #  
            if (step % step_print) == 0 and not (step == 0):
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss : {:.4f}'.format(step, len(train_dataloader), elapsed, loss.item()))
            
            # moving batch to device for faster computation(if gpu)
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            # put all the gradients to zero before forward pass
            model.zero_grad()        

            # forward pass
            outputs = model(batch_input_ids, 
                            token_type_ids=None, 
                            attention_mask=batch_input_mask, 
                            labels=batch_labels)

            loss = outputs.loss 
            logits = outputs.logits

            # training loss = sum of batch losses
            total_train_loss += loss.item()

            # calculate gradients with backward pass using loss
            loss.backward()

            # gradient clipping to a norm of 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters with calculated gradients
            optimizer.step()

            # scheduler step if it is activated ( change lr according the chosen lr scheduler)
            if scheduler:
                scheduler.step()

            if save_N_steps:
                # save the model each save_N_steps steps
                if (step % save_N_steps) == 0 and not step == 0:
                    file_name = save_path + '_epoch_'+ str(epoch_i)+'_step_'+ str(step) +'.pkl'
                    torch.save(model.state_dict(), file_name)
                    # print('  Model saved')

            # move logits and labels to cpu so we can use numpy
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()

            # flat train accuracy = sum of batch flat train accuracy
            total_train_accuracy += flat_accuracy(logits, label_ids)

        # train accuracy = flat train accuracy / num batches
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("  Train accuracy: {0:.4f}".format(avg_train_accuracy))

        # train loss = sum of batch losses / num batches
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # epoch time
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epochtook: {:}".format(training_time))

        # save model after every epoch
        if save_epoch:
            file_name = save_path + '_epoch_'+ str(epoch_i)+'.pkl'
            torch.save(model.state_dict(), file_name)

        # validation metrics after every epoch if validate = True
        if validate :
            print("")
            print("Running Validation...")

            t0 = time.time()

            # dropout and batch norm behave different during training, so activate eval mode
            model.eval()

            total_eval_accuracy, total_eval_loss = validation(model, val_dataloader, device)

            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(val_dataloader)

            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.4f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
        else:
            validation_time = 0
            avg_val_loss = 0
            avg_val_accuracy = 0

        # save training and validation metrics for one epoch
        training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Training. Accur.': avg_train_accuracy,
              'Training Time': training_time,
              'Valid. Loss': avg_val_loss,
              'Valid. Accur.': avg_val_accuracy,
              'Validation Time': validation_time
          }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return training_stats



'''' cross validation function'''
def cv_bert(input_ids, attention_masks, labels, device, PATH_DATA='./data/', model_name = 'BertWithCustomClassifier'):
    '''
    Perform cross-validation (CV) on a BERT like model, print the scores
    inputs:
        model (str): model name can be :
                -'BertWithCustomClassifier' : if you want to perform cv on the BERT model with a
                                              custom classifier instead of the default one
                -'BertForSequenceClassification' : if you want to perform cv on the default
                                                   BertForSequenceClassification model
        input_ids (tensor) : tensor containing the input ids
        attention_masks (tensor) : tensor containing the
        labels (tensor)  : tensor containing the labels
        device (str) : 'cpu' or 'gpu' to speed up training
        PATH_DATA (str) : main data path used to save the model
        
    outputs:
        None
    '''
    full_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_ds, val_ds = train_val_split(full_dataset,proportion = 0.8)
    train_dataloader = as_dataloader(train_ds, random = True, batch_size = 32) #DataLoader(train_ds, shuffle = True, batch_size = batch_size)
    val_dataloader = as_dataloader(val_ds, random = False, batch_size = 32)
    
    for epochs in [2,3]:
        for lr in [2e-5, 3e-5, 5e-5]:
            
            # initialize the model at the begging of each run
            if model_name == 'BertWithCustomClassifier' :
                model =  BertWithCustomClassifier(nb_hidden=500)
            if model_name == 'BertForSequenceClassification':
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                       num_labels = 2,
                                                                       output_attentions = False,
                                                                       output_hidden_states = False)
            # send the model parameters to the GPU
            model.to(device)
            
            # indicate which learning rate and number of epochs is currently being tested : 
            print('================================================================================')
            print('===========           lr =',lr,'                num_epochs =  ', epochs,' =============')
            print('================================================================================')
            total_steps = len(train_dataloader) * epochs # = number of batches times epochs

            optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8) # trying lr = 1e-5

            scheduler = get_linear_schedule_with_warmup(optimizer,  
                                                        num_warmup_steps = round(total_steps*0.10), 
                                                        num_training_steps = total_steps)
            
            header = 'CV_'+ model_name
            
            training_stats = train_bert_class_with_params(train_dataloader,val_dataloader,
                                              model, optimizer, scheduler,
                                              epochs, random_seed=42,
                                              device=device,
                                              PATH_DATA=PATH_DATA,
                                              save_N_steps=100000,
                                              save_epoch=False,
                                              save_path = './data/models/BERT/model',
                                              step_print=100000,
                                              validate = True,
                                              freezing = False,
                                              freez_steps = 100,
                                              frozen_epochs = 1)
            print(training_stats)