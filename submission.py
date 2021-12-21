import csv
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
def make_submission_glove(test_x, name, model):
    """
    Creates the submission when the embeddings used were glove
    Inputs :
    - test_x (ndarray) : array containing the embeddings og the tweets used for testing
    - name (string) : name given for the submission of the file
    - model (sklearn.Model) : model to use for this submission
    """
    #predict rather the tweets are positive or negative
    predictions_test = model.predict(test_x)
    
    #creates the ids of the test
    ids = np.arange(test_x.shape[1])+1
    
    #changes labels of 0 to -1 for tweets predicted as negative
    predictions_test[predictions_test==0] = -1
    create_csv_submission(ids, predictions_test, PATH_DATA + 'submissions/output_' + name + '.csv' )



def make_submission_tfidf(test_x, name, model):
    """
    Creates the submission when the embeddings used were Tf-Idf
    Inputs :
    - test_x (ndarray) : contains the embeddings of the tweets to predict
    - name (string) : name for the submission file
    - model (sklearn.Model) : model to use for the prediction
    """
    
    #predict rather the tweets are positive or negative
    predictions_test = model.predict(test_x)
    
    #creates the ids for the test
    ids = np.arange(test_x.shape[1])+1
    
    #changes labels of 0 to -1 for tweets predicted as negative
    predictions_test[predictions_test==0] = -1
    create_csv_submission(ids, predictions_test, PATH_DATA + 'submissions/output_' + name + '.csv' )