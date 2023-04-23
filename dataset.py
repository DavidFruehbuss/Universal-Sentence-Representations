from datasets import load_dataset

import nltk
from nltk.tokenize import word_tokenize

import os
import json
import collections

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SentencesDataset(Dataset):

    def __init__(self, data, new_vocab, sentence_length=82):
        super().__init__()
        self.data = data
        self.new_vocab = new_vocab
        self.sentence_length = sentence_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence1, sentence2, label = self.data[index]
        s1 = self.prepare_sentence(sentence1, self.new_vocab, self.sentence_length)
        s2 = self.prepare_sentence(sentence2, self.new_vocab, self.sentence_length)
        return s1, s2, torch.Tensor([label])

    def prepare_sentence(self, data_batch, new_vocab, sentence_length):
      '''
      prepares data for training
      '''
      x = torch.zeros(sentence_length)
      s_length = len(data_batch)
      for i, w in enumerate(data_batch):
        if w in new_vocab.keys():
          x[i] = torch.LongTensor([new_vocab[w]])
        else:
          x[i] = torch.LongTensor([0])
      return (x, s_length)

def get_data(save_path=None, glove_path=None):

    '''
    creates or loads the processed data and returns dataloaders, emveddings and vocab
    '''

    if save_path == None:
        save_path = './data/'

    if os.path.isfile(save_path + 'new_data'):

        print('Loading data')

        # load datasets
        with open(save_path + 'new_data', 'r') as f:
            data = json.load(f)
        # load alligned glove embeddings
        vectors = torch.load(save_path + 'vectors')
        # load new vocab
        with open(save_path + 'vocab', 'r') as f:
            new_vocab = json.load(f)

    else:

        print('Creating data')

        data, vectors, new_vocab = create_data(glove_path)

    # create new Dataset objects
    new_train_data, new_val_data, new_test_data = data
    train_dataset = SentencesDataset(new_train_data, new_vocab)
    val_dataset = SentencesDataset(new_val_data, new_vocab)
    test_dataset = SentencesDataset(new_test_data, new_vocab)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, vectors, new_vocab

def create_data(glove_path=None):

    '''
    1. load dataset from Huggingface website
    2. filter dataset for -1 labels
    2. basic preprocessing (tokenization, lower_caseing)
    3. changing dataset format for creating SentenceDataset object
    4. creating the vocabulary
    5. load Glove embeddings and allign with dataset
    6. save data splits, vocab, and embedding matrix
    '''

    # load datasets
    train_data_raw = load_dataset('snli', split='train')
    val_data_raw = load_dataset('snli', split='validation')
    test_data_raw = load_dataset('snli', split='test')

    # filter dataset
    train_data_filterd = train_data_raw.filter(lambda example: example['label'] != -1)
    val_data_filterd = val_data_raw.filter(lambda example: example['label'] != -1)
    test_data_filterd = test_data_raw.filter(lambda example: example['label'] != -1)

    print(train_data_filterd)
    print(val_data_filterd)
    print(test_data_filterd)

    # preprocessing
    nltk.download('punkt')
    train_data = preprocess(train_data_filterd)
    val_data = preprocess(val_data_filterd)
    test_data = preprocess(test_data_filterd)

    # changing dataset format for creating SentenceDataset object
    premise_train, hypothesis_train, label_train = train_data
    premise_val, hypothesis_val, label_val = val_data
    premise_test, hypothesis_test, label_test = test_data

    new_train_data = []
    for i in range(len(premise_train)):
        new_train_data.append((premise_train[i], hypothesis_train[i], label_train[i]))
    print('Amount of training data:', len(new_train_data))

    new_val_data = []
    for i in range(len(premise_val)):
        new_val_data.append((premise_val[i], hypothesis_val[i], label_val[i]))
    print('Amount of validation data:', len(new_val_data))

    new_test_data = []
    for i in range(len(premise_test)):
        new_test_data.append((premise_test[i], hypothesis_test[i], label_test[i]))
    print('Amount of test data:', len(new_test_data))

    # create a vocabulary and find max sentence length
    vocab_dict = {}
    max_len = 0

    premise_train, hypothesis_train, _ = train_data
    premise_val, hypothesis_val, _ = val_data
    premise_test, hypothesis_test, _ = test_data

    data = [premise_train, hypothesis_train, premise_val, hypothesis_val, premise_test, hypothesis_test]
    for d in data:
        for s in d:
            if len(s) > max_len: max_len = len(s)
            for w in s:
                vocab_dict.setdefault(w, 0)
                vocab_dict[w] += 1

    vocab_size = len(vocab_dict)
    print('Vocabulary length:', vocab_size)

    # load and align glove embeddings
    if glove_path == None:
        glove_path = './data/glove.840B.300d.txt'
    else:
        glove_path = glove_path

    embeddings_dict = {}

    with open(glove_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in vocab_dict.keys():
                if len(values) != 301: continue
                vector = torch.FloatTensor([float(i) for i in values[1:]])
                embeddings_dict[word] = vector

    new_vocab = {}
    idx = 2
    for i, key in enumerate(embeddings_dict.keys()):
        new_vocab[key] = idx
        idx += 1
    print('Vocabulary length after alligning:', len(new_vocab))
    print('max sentence length:', max_len)

    # create Embedding matrix with only words that have pretrained embeddings
    vectors = torch.zeros(len(embeddings_dict) + 2, 300)
    # pad and unk token take position 0, 1
    for i, key in enumerate(embeddings_dict.keys()):
        vectors[i+2] = embeddings_dict[key]

    # save data splits, vocab, and embedding matrix
    new_data = [new_train_data, new_val_data, new_test_data]
    save_path = './data/'

    # save datasets
    with open(save_path + 'new_data', 'w') as f:
                json.dump(new_data, f)
    # save alligned glove embeddings
    torch.save(vectors, save_path + 'vectors')
    # save new vocab
    with open(save_path + 'vocab', 'w') as f:
                json.dump(new_vocab, f)

    return new_data, vectors, new_vocab

def preprocess(data):

    '''
    apply tokenization and lowercaseing as preprocessing steps
    '''

    data_token_p = [word_tokenize(s) for s in data['premise']]
    data_token_h = [word_tokenize(s) for s in data['hypothesis']]

    data_lower_p = [[w.lower() for w in sentence] for sentence in data_token_p]
    data_lower_h = [[w.lower() for w in sentence] for sentence in data_token_h]

    return [data_lower_p, data_lower_h, data['label']]
