'''
MIT License

Copyright (c) 2017 Mat Leonard

'''
import re

import torch.nn as torch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import syllables

from lstm.src.model.utils import one_hot_encode

class CharRNN(nn.Module):
    """
    A class for a Char RNN using pytorch

    Arguemnts:
    ----------
    tokens = the number of tokens(chars) in the data
    n_steps = the batch size
    n_hidden = the size of the hidden layers 
    n_layers = the number of hidden layers
    drop_prob = the drop out probablity 
    lr = learning rate 
    """
    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))
        
        self.init_weights()
        
    def forward(self, x, hc):
        ''' Forward pass through the network '''
        
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        
        # Stack up LSTM outputs
        x = x.reshape(x.size()[0]*x.size()[1], self.n_hidden)
        
        x = self.fc(x)
        
        return x, (h, c)
    
    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.
        
            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()
        
        if h is None:
            h = self.init_hidden(1)
        
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = Variable(torch.from_numpy(x), volatile=True)
        if cuda:
            inputs = inputs.cuda()
        
        h = tuple([Variable(each.data, volatile=True) for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out).data
        if cuda:
            p = p.cpu()
        
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
            
        return self.int2char[char], h
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()),
                Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()))

def save_model(model, filename='rnn.ckpt'):
    """Saves the model."""

    checkpoint = {'n_hidden': model.n_hidden,
                  'n_layers': model.n_layers,
                  'state_dict': model.state_dict(),
                  'tokens': model.chars}
    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)

def load_model(filename):
    """Loads a model from a given file."""

    with open(filename, 'rb') as f:
        checkpoint = torch.load(f)

    model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

def sample(model, size, prime='The', top_k=None, cuda=False):
    """ Sample characters from the model."""

    # Sees if cuda is aviable
    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    # Iterates through the primers and creates poems based on the primer
    for ch in prime:
        char, h = model.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    for _ in range(size):
        char, h = model.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


def normalize_haiku_text(haiku):
    haiku = haiku.replace("\n", " ")
    haiku = " ".join(haiku.split())
    haiku = haiku.split(" ")
    haiku = [word if re.findall("I", word) else word.lower() for word in haiku]
    return(haiku)

def split_haiku_by_line(haiku):
    haiku_generator = (word for word in haiku)
    haiku_with_line_breaks = ""
    running_syllable_count = 0
    line_number = 0
    while line_number < 3:
        word = next(haiku_generator)
        running_syllable_count += syllables.estimate(word)
        if running_syllable_count >= 5 and line_number in (0, 2):
            haiku_with_line_breaks += f"{word}\n"
            line_number += 1
            running_syllable_count = 0
        elif running_syllable_count >= 7 and line_number == 1:
            haiku_with_line_breaks += f"{word}\n"
            line_number += 1
            running_syllable_count = 0
        else:
            haiku_with_line_breaks += f"{word} "
    return haiku_with_line_breaks