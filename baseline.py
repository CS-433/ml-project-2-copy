import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neural_network import MLPClassifier



'''Handling of the features'''


def df_to_numpy_features(train_pos_embeddings,train_neg_embeddings):
    """
    Assemble the embeddings for positive and negative tweets and convert them to numpy arrays  
    Inputs:
    - train_pos_embeddings (DataFrame) : dataframe containing the glove embeddings of the positive tweets 
    - train_neg_embeddings (ndarray) : dataframe containing the glove embeddings of the negative tweets
    Outputs:    
    - x (ndarray) : positive and negative tweets' embeddings merged into one numpy array
    - y (ndarray) : labels of the corresponding data point 1 for positive 0 for negative
    """
    x_1 = train_pos_embeddings.to_numpy(copy=True)
    x_2 = train_neg_embeddings.to_numpy(copy=True)

    # label vectors, 1 for positive, 0 for negative
    y_1 = np.ones(len(x_1))
    y_2 = np.zeros(len(x_2))

    #merge the tweets
    x = np.vstack((x_1,x_2))
    y = np.hstack((y_1,y_2)) 
    return x, y



def standardize_cols(x, mean_x=None, std_x=None):
    """
    Standardizes the x passed  
    Inputs:
    - x (ndarray) : dataset to standardize
    - mean_x (ndarray) : by default to None, if an array is passed as an argument, standardize the array using those means
    - std_x (ndarray) : by default to None, if an array is passed as an argument, standardize the array using those standard 
                        deviations
    Outputs:    
    - x (ndarray) : dataset standardized
    - mean_x (ndarray) : means of the features pre-standardization
    - std_x (ndarray) : standard deviatinos of the features pre-standardization
    """
    
    # checks if mean_x is None and if it is, calculates it as well a std_x
    if np.logical_not(np.all(mean_x)):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
    # standardizes
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x


'''Naive Bayes'''

def naive_bayes_cv(x, y, splits=5, glove=True):
    """
    Performs cross validation with naive bayes model 
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - splits (int) : number of folds when doing cross-validation
    - glove (boolean) : by default to True, if True the embeddings are glove, if False they are TF-IDF embeddings
    Outputs:    
    - cvb (list) : list containing the accuracy of each fold
    - cvb.mean() (float) : mean of the accuracies
    - cvb.std() (float= : standard deviation of the accuracies
    """
    
    #separates the data into different random folds
    k_fold = KFold(n_splits=splits, shuffle=True, random_state=0)
    
    if glove :
        #if the embeddings are glove we used the method from sklearn GaussianNB() because the x passed is a numpy array
        gnb = GaussianNB()
        #SKlearn method that obtains the cross-validation score for all the folds
        cvb = cross_val_score(gnb, x, y, cv=k_fold, n_jobs=-1)
        print(cvb)
        print("Mean of accuracy for Naive Bayes " + str(cvb.mean()))
        print("Std of accuracy for Naive Bayes " + str(cvb.std()))
        return cvb, cvb.mean(), cvb.std()
    else :
        #if the embeddings are done using TF-IDF we used the method from sklearn MultinomialNB() because the matrix passed as X
        #is sparse
        gnb = MultinomialNB()
        #SKlearn method that obtains the cross-validation score for all the folds
        cvb = cross_val_score(gnb, x, y, cv=k_fold, n_jobs=-1)
        print(cvb)
        print("Mean of accuracy for Naive Bayes " + str(cvb.mean()))
        print("Std of accuracy for Naive Bayes " + str(cvb.std()))
        return cvb, cvb.mean(), cvb.std()

def naive_bayes(x, y, glove=True):
    """
    Trains a naive Bayes model
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - glove (boolean) : by default to True, if True the embeddings are glove, if False they are TF-IDF embeddings
    Outputs:    
    - gnb (model): returns the naive Bayes model
    """
    
    # https://www.analyticsvidhya.com/blog/2021/01/gaussian-naive-bayes-with-hyperpameter-tuning/

    if glove :
        #if the embeddings are glove we used the method from sklearn GaussianNB() because the x passed is a numpy array
        gnb = GaussianNB()
        gnb.fit(x, y)
        return gnb
    else :
        #if the embeddings are done using TF-IDF we used the method from sklearn MultinomialNB() because the matrix passed as X
        #is sparse
        gnb = MultinomialNB()
        gnb.fit(x, y)
        return gnb




'''Logistic Regression'''

def logistic_regression_cv(x, y, solvers = ['lbfgs', 'saga'], penalty = ['l2'], c_values = [100, 10, 1.0, 0.1, 0.01], splits = 5):
    """
    Performs grid_search and cross_validation on a logistic regression model
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - solvers (list) : list containing the different solvers on which to perform grid_search on
    - penalty (list) : list containing the different penalties on which to perform grid_search on
    - c_values (list) : list containing the inverse of the regularization strength to perform grid_search on
    - splits (int) : number of folds for the cross-validation
    Outputs:    
    - means (list) : list containing the means for each combination of hyperparameter
    - stds (list) : list containing the standard deviations for each combination of hyperparameter
    - params(list) : list containing the combination of hyperparameters corresponding to each mean and std
    """ 

    #https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
    #https://scikit-learn.org/stable/modules/linear_model.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
    #https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
    model = LogisticRegression()
    #creates the grid for the grid-search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    
    #divides the data into folds
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=3, random_state=1)
    
    #does the grid-search and cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    
    return means, stds, params


def logistic_regression(x, y, solver='saga', penalty='l2', c_value=1.0):
    """
    Trains a logistic regression model and returns it
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - solver (string) : solver to use to train the model
    - penalty (string) : penalty to use to train the model
    - c_value (int) : inverse of the regularization strength to use to train the model
    Outputs:    
    - model (sklearn.LogisticRegression): returns the Logistic regression model trained
    """
    #https://scikit-learn.org/stable/modules/linear_model.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    #https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389
    
    #creates the model
    model = LogisticRegression(solver = solver, penalty = penalty, C=c_value)
    
    #fits the model
    model.fit(x, y)
    return model 



'''Multi-Layer Perceptron'''


def mlp_cv(x, y, solvers =['adam'], lrs = [0.1, 0.01,0.001], activations = ['tanh','relu','logistic'], max_iters=[5,50], splits = 5):
    """
    Performs grid-search and cross-validation on the multi-layer perceptron
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - solvers (list) : solvers on which to perform grid-search 
    - lrs (list) : learning rates on which to perform grid-search 
    - activations (list) : activation function on which to perform grid-search 
    - max_iters (list) : maximum number of iterations on which to perform grid-search
    - splits (int) : number of folds for cross-validation
    Outputs:    
    - means (list) : mean of the accuracies for each combination of hyperparameters
    - stds (list) : standard deviations of the accuracies for each combination of hyperparameters
    - params (list) : list containing the different combination of hyperparameters for every mean and standard deviation
    """
    
    #creates the model 
    # early stopping take 10% of dataset if val accuracy doesn't change after 2 epochs training is stopped
    model = MLPClassifier(early_stopping=True, n_iter_no_change=2)
    
    #creates the grid of parameters
    grid = dict(solver=solvers,activation=activations,learning_rate_init=lrs,max_iter=max_iters)
    
    #creates the folds for cross-validation
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=1, random_state=1)
    
    #performs the grid-search and cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0,verbose=True)
    grid_result = grid_search.fit(x, y)
    
    #summarize the results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
        
    return means, stds, params

def mlp(x, y, solver='adam', lr=0.001, act='tanh', max_iters=5):
    
    """
    Trains a multi-layer perceptron
    Inputs:
    - x (ndarray) : array containing the datapoints
    - y (ndarray) : array containing the labels of the datapoints
    - solver (string) : solver to use for the training of the model
    - lr (int) : learning rate to use for the training of the model
    - act (string) : activation function to use for the training of the model 
    - max_iters (int) : maximum number of iterations to use for the training of the model
    Outputs:    
    - mlp (SKlearn.MLPClassifier) : multi-layer perceptron trained
    """
    #https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps
    mlp = MLPClassifier(random_state=1, max_iter=max_iters, solver = solver,
                    activation = act, learning_rate_init = lr,
                    early_stopping=True, n_iter_no_change=2)
    mlp.fit(x, y)
    return mlp