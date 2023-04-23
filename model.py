import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy


class USC(nn.Module):
  '''
  Input shape: [batch_size, 3, sentence length, embedding]
  '''

    
  def __init__(
      self,
      encoder,
      vocab_size,
      sentence_length,
      vector_embeddings,
      embedding_dim=300,
  ):
      super().__init__()

      if encoder == 'baseline':
        self.encoder_p = baseline(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, embedding_dim=300,)
        self.encoder_h = baseline(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, embedding_dim=300,)
        emb_size = 300
      elif encoder == 'lstm':
        self.encoder_p = LSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        self.encoder_h = LSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        emb_size = 2048
      elif encoder == 'bilstm':
        self.encoder_p = BiLSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        self.encoder_h = BiLSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        emb_size = 4096
      elif encoder == 'poollstm':
        self.encoder_p = PoolBiLSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        self.encoder_h = PoolBiLSTM(vocab_size=vocab_size, sentence_length=sentence_length, vector_embeddings=vector_embeddings, hidden_size=2048, embedding_dim=300,)
        emb_size = 4096

      self.mlp = nn.Sequential(
          nn.Linear(4*emb_size,512),
          nn.ReLU(),
          nn.Linear(512,3),
          nn.Softmax(dim=-1),
      )



  def forward(self, h, h_len, p, p_len):

      h = self.encoder_h(h, h_len)
      p = self.encoder_p(p, p_len)

      us = torch.cat((h,p,torch.abs(h-p), h*p), dim=1)

      usc = self.mlp(us)

      return usc

class baseline(nn.Module):

    '''
    Input shape: [batch_size, sentence length, embedding]
    
    '''

    def __init__(
        self,
        vocab_size,
        sentence_length,
        vector_embeddings,
        embedding_dim=300,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(vector_embeddings)
        self.embedding.weight.requires_grad = False


    def forward(self, x, x_len):
        
        x = x.type(torch.int64)
        x = self.embedding(x)
        x = torch.mean(x, dim=1)

        return x

class LSTM(nn.Module):

    '''
    Input shape: [batch_size, sentence length, embedding]
    
    '''

    def __init__(
        self,
        vocab_size,
        sentence_length,
        vector_embeddings,
        hidden_size=2048,
        embedding_dim=300,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(vector_embeddings)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)


    def forward(self, x, x_len):
        
        max_len = x.shape[1]
        x = x.type(torch.int64)
        x_len = x_len.type(torch.int64).to('cpu')
        x = self.embedding(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(x_packed)
        output_unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_len)[0]

        out = output_unpacked[:,0,:]
        return out

class BiLSTM(nn.Module):

    '''
    Input shape: [batch_size, sentence length, embedding]
    
    '''

    def __init__(
        self,
        vocab_size,
        sentence_length,
        vector_embeddings,
        hidden_size=4096,
        embedding_dim=300,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(vector_embeddings)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True)


    def forward(self, x, x_len):
        
        max_len = x.shape[1]
        x = x.type(torch.int64)
        x_len = x_len.type(torch.int64).to('cpu')
        x = self.embedding(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(x_packed)
        output_unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_len)[0]

        out = output_unpacked[:,0,:]
        return out


class PoolBiLSTM(nn.Module):

    '''
    Input shape: [batch_size, sentence length, embedding]
    
    '''

    def __init__(
        self,
        vocab_size,
        sentence_length,
        vector_embeddings,
        hidden_size=4096,
        embedding_dim=300,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(vector_embeddings)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True)


    def forward(self, x, x_len):
        
        max_len = x.shape[1]
        x = x.type(torch.int64)
        x_len = x_len.type(torch.int64).to('cpu')
        x = self.embedding(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(x_packed)
        output_unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_len)[0]

        out, _ = torch.max(output_unpacked, dim=1)

        return out

