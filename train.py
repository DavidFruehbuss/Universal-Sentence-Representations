import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_data
from model import USC

import wandb
import argparse

def train(
    model, 
    train_loader,
    val_loader,
    test_loader,
    learning_rate,
    checkpoint_path,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.to(device)

    top_acc = 0
    epoch = 0

    while learning_rate > 1e-5:

      if epoch == 15:
        print('Early stoping')
        return

      epoch += 1
      epoch_loss = 0
      epoch_accuracy = 0
      accuracy_len = 0

      for data_batch_p, data_batch_h, label in train_loader:

          p, p_len = data_batch_p
          h, h_len = data_batch_h

          p = p.to(device)
          h = h.to(device)
          p_len = p_len.to(device)
          h_len = h_len.to(device)

          prediction = model.forward(p, p_len, h, h_len)
          label = torch.nn.functional.one_hot(label.type(torch.LongTensor))
          label = label.squeeze(1).type(torch.FloatTensor)
          label = label.to(device)

          loss = loss_fn(prediction, label)

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          epoch_accuracy += torch.sum(torch.argmax(prediction, dim=1) - torch.argmax(label, dim=1) == 0)
          accuracy_len += len(prediction)
          epoch_loss += loss

      epoch_accuracy = epoch_accuracy / accuracy_len
      val_acc = eval(val_loader)
      test_acc = eval(test_loader)

      print(f'epoch: {epoch} \n')
      print(f'Training loss: {epoch_loss}')
      print(f'Training accuracy: {epoch_accuracy}')
      print(f'Validation accuracy: {val_acc}')
      print(f'Test accuracy: {val_acc}')

      wandb.log({'epoch': epoch})
      wandb.log({'Training loss': epoch_loss})
      wandb.log({'Training accuracy': epoch_accuracy})
      wandb.log({'Validation accuracy': val_acc})
      wandb.log({'Test accuracy': test_acc})

      if val_acc > top_acc:
        top_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        # papers learning rate reduction
        learning_rate = learning_rate / 5

      # papers learning rate decay (each epoch)
      learning_rate = learning_rate * 0.99
      optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print('Training complete')

def eval(model, loader):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model.to(device)

  val_accuracy = 0
  val_accuracy_len = 0

  for data_batch_p, data_batch_h, label in loader:

      p, p_len = data_batch_p
      h, h_len = data_batch_h

      p = p.to(device)
      h = h.to(device)
      p_len = p_len.to(device)
      h_len = h_len.to(device)

      prediction = model.forward(p, p_len, h, h_len)
      label = torch.nn.functional.one_hot(label.type(torch.LongTensor))
      label = label.squeeze(1).type(torch.FloatTensor)
      label = label.to(device)

      val_accuracy += torch.sum(torch.argmax(prediction, dim=1) - torch.argmax(label, dim=1) == 0)
      val_accuracy_len += len(prediction)

  val_accuracy = val_accuracy / val_accuracy_len

  return val_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', default='baseline', type=str,
                        help='Encoder model to use')
    parser.add_argument('--glove_path', default=None, type=str,
                        help='glove_path')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='checkpoint_path')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning_rate')

    args = parser.parse_args()

    config = {
    'model': args.model,
    'sentence_length': 82,
    'learning_rate': 0.1

    

    }

    wandb.init(project="Universal Sentence Representaion", entity="inspired-minds", name=config['model'], config=config)
    print('Hyperparameters: ', wandb.config)

    train_loader, val_loader, test_loader, vectors, new_vocab = get_data(save_path=None, glove_path=args.glove_path)

    model = USC(encoder=wandb.config['model'], vocab_size=len(vectors), sentence_length=wandb.config['sentence_length'], vector_embeddings=vectors, embedding_dim=300,)
    model_name = wandb.config['model']

    if args.checkpoint_path == None:
      checkpoint_path = f'./saves/{model_name}/checkpoint.pth'
    else:
      checkpoint_path = args.checkpoint_path

    train(model, train_loader, val_loader, test_loader, wandb.config['learning_rate'], checkpoint_path)

