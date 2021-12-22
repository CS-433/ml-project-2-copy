# ml-project-2-copy

## Description
This project is part of the EPFL Machine Learning course for fall 2021 and was implemented by Younes Moussaif, Clémence Barsi and Pauline Conti.

The aim of this project is to predict implement machine learning models able to predict if a tweet used to contain a ":)" or ":(" smiley face. The training data available contains 2.5 Million tweets that the models implemented need to correctly classify as positive or negative. To obtain our test accuracy, we uploaded our submissions to AIcrowd.


## Versions and Libraries
The python libraries and versions used for this project are listed below:
- python = 3.7.11
- scipy = 1.7.1
- pandas = 1.3.4
- numpy = 1.21.2
- matplotlib = 3.5.0
- pytorch = 1.10.1
  - torchvision = 0.11.2
  - torchaudio = 0.10.1
  - cudatoolkit = 10.2
- scikit-learn = 1.0.1
- jupyterlab = 3.2.1
  - nb_conda_kernels
- nltk = 4.62.3
- transformers = 4.14.1

In order to run the notebooks, run the following command when the environnement with the above dependencies is active:
```conda install -c conda-forge ipywidgets```

## How to Use
To run our best performing model, the script run.py must be ran having the same structure as in the github file meaning :
  - in the folder data/models/BERT both files :
          - best_submission_bert_custom.pkl
          - best_submission_bert.pkl
  - in the folder data/twitter-datasets/
          - test_data.csv
          - train_pos.csv
          - train_neg.csv
          - train_pos_full.csv
          - train_neg_full.csv

Since the two .pkl file are too big for github we uploaded them on an external google drive here : [drive link](https://drive.google.com/drive/folders/1hAsNuEbsmkgaBuEapGjLKX5sa76Bc809?usp=sharing)
This will create a csv output file containing our predictions.

If you want to train one of our BERT models we recommend you use google COLAB, and a use a GPU (we used it with P100 GPU), more information on how to do that is is the notebook `BERT_models.ipynb`


On a GPU it takes approximately 1 minute to run, on a CPU 10 minutes.

To reproduce our other results, you can start by openning `preprocessing_embedding_baseline_example` and following the next step (all the next steps are implemented in the notebook except for placing the large file `glove-twitter-25.gz` in the right place)
- additionally to the packages you must install at the beggingin of `BERT_models.ipynb` install gensim using `pip install gensim`
- first preprocess the tweets using the function `preprocessing` from `preprocessing.py` ( on train_pos and train_neg, some additional files required to run the functions are provided in our repo)
- Transform the tweets into number vectors (embed them) using functions in `embeddings.py` :
  - Glove :
    - make sure you have download all the files in data directory 
      - `glove-twitter-25.gz` was too large to host on github so we host on google drive : [link](https://drive.google.com/file/d/1aWogx_xnky0c9gE_avHuRr__8zZNO1Us/view?usp=sharing) 
    - load the model with `load_glove_model`
    - use clean `clean_cols` to remove unused columns
    - use `df_to_numpy_features` to obtain a numpy matrix containing the features and one that contains the labels
    - normalize the features using `standardize_cols`
  - TF-IDF :
    - use `tf_idf_embedding` to produce the sparse embeddings then `add_label_tfidf` to get the corresponding labels
- Run one of the models or crossvalidation functions from `baseline.py`
- If you want to make a submission use the corresponding submission function in `submission.py`

## Contents of the Project
- ```BERT_models.ipynb``` contains the pipeline we used for the BERT models
- ```EDA.ipynb``` contains the code we used to generate Figure 1 from the report
- `graph_BERT.ipynb` contains the code to produce the Figure 2 from the report
- ```baseline.py``` contains the functions used to train and fit our baseline models
- ```embeddings.py``` contains the functions used to obtain the embeddings for the models
- ```helpers.py``` contains helper functions
- ```helper_bert.py``` contains the helpers needed for the BERT models
- ```models_bert.py``` contains the funtions and implementation of the BERT models
- ```preprocessing.py``` contains the functions used to preprocess our data before embedding
- ```preprocessing_bert.py``` contains the functions used to preprocess the data before BERT
- ```submission.py``` allows to generate the files needed to do a sumbission on Ai-Crowd
- ```train_bert.py``` contains the functions performing the training of the BERT models.
