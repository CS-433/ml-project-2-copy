import random
import numpy as np
import torch
import time
from tqdm import tqdm
#from .helper_bert import *
from helpers_bert import *
import time
import datetime

def validation(model, val_dataloader, device, total_eval_accuracy = 0, total_eval_loss = 0):
    '''
    returns validation accuracy and loss 
    
    inputs:
    model (nn.Module)
    val_dataloader (Dataloader)
    device (string)
    total_eval_accuracy ()
    total_eval_loss ()
    
    outputs:
    total_eval_accuracy () validation accuracy
    total_eval_loss () validation loss
    '''
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
                                 txt_header = 'random',step_print=100,
                                 validate = True,
                                 freezing = False,
                                 freez_steps = 100,
                                 frozen_epochs = 1):
    '''
    
    
    
    Inputs: 
    train_dataloader
    val_dataloader (Dataloader or bool)
    model (nn.module)
    optimizer 
    scheduler
    epochs
    random_seed
    device
    PATH_DATA
    save_N_steps (bool)
    save_epoch (bool)
    txt_header = 'random'
    step_print = 100
    validate (bool)
    save_N_steps=False : don't save model every N steps
    save_N_steps=10000 : save model every 10000 steps
    save_epoch (bool) : save the model to drive at the end of every epoch
    freezing (bool)
    freez_steps (int)
    frozen_epochs (bool)
    
    Outputs:
    
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

        # begging of each epoch timer
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
                    file_name = PATH_DATA+'models/BERT/' + txt_header + '_epoch_'+ str(epoch_i)+'_step_'+ str(step) +'.pkl'
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
        file_name = PATH_DATA+'models/BERT/' + txt_header + '_epoch_'+ str(epoch_i)+'.pkl'
        torch.save(model.state_dict(), file_name)

    # validation metrics after every epoch if validate = True
    if validate :
        print("")
        print("Running Validation...")

        t0 = time.time()

        # dropout and batch norm behave different during training, so activate eval mode
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        total_eval_accuracy, total_eval_loss = validation(model, val_dataloader, device, total_eval_accuracy, total_eval_loss)

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
        
    # save training and validation metric for one epoch
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