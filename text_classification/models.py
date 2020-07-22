# models.py - define model architectures.

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class ModelLSTM(nn.Module):
    
    def __init__(self, embedding_dim, vocab_size, hidden_dim, dropout_p,
    			 num_classes, stacked_layers, padding_idx=0):
        super(ModelLSTM, self).__init__()

         # Initialize embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(
                pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx, _weight=pretrained_embeddings)

        # Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False
        
        self.word_embeddings = nn.Embedding(num_embeddings = vocab_size, 
        									embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_dim= hidden_dim, 
        					batch_first = True, num_layers = stacked_layers, 
        					dropout_p = dropout_p)
        self.linear = nn.Linear(in_features = hidden_dim, out_features=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x_batch):
        len_list = list(map(len, x_batch))
        
        padded_batch = pad_sequence(x_batch, batch_first=True)
        embeds = self.word_embeddings(padded_batch)
        pack_embeds = pack_padded_sequence(embeds, lengths=len_list, 
        								   batch_first=True, 
        								   enforce_sorted=False)
        
        rnn_out, (rnn_h, rnn_c) = self.lstm(pack_embeds)
        linear_out = self.linear(self.tanh(rnn_h))
        y_out = linear_out[-1]
        
        return y_out