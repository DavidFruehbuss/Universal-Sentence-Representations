from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import sklearn

import torch
import argparse
import json

from model import USC
from dataset import get_data
from train import eval

def prepare(params, samples):
    return

def prepare_sentence_batch(data_batch, new_vocab, sentence_length):
    '''
    prepares data for training
    '''
    sentence_batch = []
    sentence_l = []
    for s in data_batch:
        x = torch.zeros(sentence_length)
        s_length = len(s)
        for i, w in enumerate(s):
          if w in new_vocab.keys():
            x[i] = torch.LongTensor([new_vocab[w]])
          else:
            x[i] = torch.LongTensor([0])
        sentence_batch.append(x)
        sentence_l.append(s_length)
    sentence_batch = np.vstack(sentence_batch)
    sentence_l = np.vstack(sentence_l)

    return torch.from_numpy(sentence_batch), torch.from_numpy(sentence_l)

def batcher(params, batch):

    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    # prepare sentences
    with torch.no_grad():
      x, x_len = prepare_sentence_batch(batch, new_vocab, params.sentence_length)
      x = x.to(device)
      x_len = x_len.squeeze(1).to(device)

      # We only really need the enocder to get the sentence representation
      embeddings = model.encoder_h(x, x_len)
      
      embeddings = embeddings.to('cpu')
      embeddings = embeddings.detach()
    return embeddings.numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', default='baseline', type=str,
                        help='Encoder model to use')
    parser.add_argument('--glove_path', default=None, type=str,
                        help='glove_path')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='checkpoint_path')
    parser.add_argument('--result_path', default='./results/', type=str,
                        help='result_path')

    args = parser.parse_args()

    config = {
    'model': args.model,
    'sentence_length': 150,
    'learning_rate': 0.1
    }

    # path to the NLP datasets 
    PATH_TO_DATA = './data'

    import senteval

    # start by loading the model and doing SNLI task

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint_path == None:
        checkpoint_path = f'./saves/{args.model}/checkpoint.pth'
    else:
        checkpoint_path = args.checkpoint_path

    config = {
        'model': 'baseline',
        'sentence_length': 150,

    }

    train_loader, val_loader, test_loader, vectors, new_vocab = get_data(save_path=None, glove_path=None)
    model = USC(encoder=args.model, vocab_size=len(vectors), sentence_length=config['sentence_length'], vector_embeddings=vectors, embedding_dim=300,)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    test_acc = eval(model, test_loader)
    val_acc = eval(model, val_loader)

    print(test_acc)
    print(val_acc)

    ## SentEval

    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10, 'sentence_length': config['sentence_length']}

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # SICKRelatedness (Sick-R) needs torch cuda to work (even when using logistic regression), 
    # but STS14 (semantic textual similarity) is a similar type of semantic task
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                    'MRPC', 'SICKEntailment', 'STS14']
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results)

    # safe results
    result_path = args.result_path
    with open(result_path + args.model, 'w') as f:
        json.dump(results, f)


