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
    predictions_test = model.predict(test_x)
    ids = test_data_embeddings.index.array.to_numpy()
    ids = ids + 1
    predictions_test[predictions_test==0] = -1
    create_csv_submission(ids, predictions_test, PATH_DATA + 'submissions/output_' + name + '.csv' )



def make_submission_tfidf(test_x, name, model):
    predictions_test = model.predict(test_x)
    ids = np.arange(test_x.shape[1])+1
    predictions_test[predictions_test==0] = -1
    create_csv_submission(ids, predictions_test, PATH_DATA + 'submissions/output_' + name + '.csv' )