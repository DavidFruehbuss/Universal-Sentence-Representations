from datasets import load_dataset

import nltk
from nltk.tokenize import word_torkenizer
from nltk.stem.porter import *

def load_data():

    '''
    load dataset from Huggingface website
    '''

    train_data_raw = load_dataset('snli', split='train')
    val_data_raw = load_dataset('snli', split='train')
    test_data_raw = load_dataset('snli', split='train')

    # preprocessing
    nltk.download('punkt')
    train_data = preprocess(train_data_raw)
    val_data = preprocess(val_data_raw)
    test_data = preprocess(test_data_raw)
    



    return train_data, val_data, test_data

def preprocess(data):

    '''
    apply tokenization and lowercaseing as preprocessing steps
    '''

    data_token_p = [word_tokenize(s) for s in train_data['premise']]
    data_token_h = [word_tokenize(s) for s in train_data['hypothesis']]

    data_lower_p = [[w.lower() for w in sentence] for sentence in data_token_p]
    data_lower_h = [[w.lower() for w in sentence] for sentence in data_token_h]


    # data.filter(label != -1)

    return [data_lower_p, data_lower_h, train_data['label']]
