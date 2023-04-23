import numpy as np
import torch

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

