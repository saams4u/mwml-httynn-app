# models.py - define model architectures.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class ModelLSTM(nn.Module):
    
    def __init__(self, embedding_dim, vocab_size, hidden_dim, stacked_layers, 
    			 dropout_p, num_classes, pretrained_embeddings=None, 
    			 freeze_embeddings=False, padding_idx=0):
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
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, 
		                    batch_first = True, num_layers = stacked_layers, 
		                    dropout = dropout_p)
        self.linear = nn.Linear(in_features = hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_batch):
        len_list = list(map(len, x_batch))
        
        padded_batch = pad_sequence(x_batch, batch_first=True)
        embeds = self.word_embeddings(padded_batch)
        pack_embeds = pack_padded_sequence(embeds, lengths=len_list, 
                           batch_first=True, 
                           enforce_sorted=False)
        
        rnn_out, (rnn_h, rnn_c) = self.lstm(pack_embeds)
        linear_out = self.linear(self.sigmoid(rnn_h))
        y_out = linear_out[-1]
        
        return y_out


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_filters, filter_sizes,
                 hidden_dim, dropout_p, num_classes, pretrained_embeddings=None,
                 freeze_embeddings=False, padding_idx=0):
        super(TextCNN, self).__init__()

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

        # Conv weights
        self.filter_sizes = filter_sizes
        self.conv = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim,
                       out_channels=num_filters,
                       kernel_size=f) for f in filter_sizes])

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_filters*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, channel_first=False):

        # Embed
        x_in = self.embeddings(x_in)
        if not channel_first:
            x_in = x_in.transpose(1, 2)  # (N, channels, sequence length)

        z = []
        conv_outputs = []  # for interpretability
        max_seq_len = x_in.shape[2]
        for i, f in enumerate(self.filter_sizes):

            # `SAME` padding
            padding_left = int(
                (self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2)
            padding_right = int(math.ceil(
                (self.conv[i].stride[0]*(max_seq_len-1) - max_seq_len + self.filter_sizes[i])/2))

            # Conv
            _z = self.conv[i](F.pad(x_in, (padding_left, padding_right)))
            conv_outputs.append(_z)

            # Pool
            _z = F.max_pool1d(_z, _z.size(2)).squeeze(2)
            z.append(_z)

        # Concat outputs
        z = torch.cat(z, 1)

        # FC
        z = self.fc1(z)
        z = self.dropout(z)
        logits = self.fc2(z)

        return conv_outputs, logits


 class ModelCnnRnn(nn.Module):

 	def __init__(self, embedding_dim, vocab_size, hidden_dim, num_classes):
 		super(ModelCnnRnn, self).__init__()

 		self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, 
 											embedding_dim=embedding_dim)
 		
 		self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, 
 							   stride=1, padding=1)

 		self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, 
 							   padding=1)

 		self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, 
 							   padding=1)

 		self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)

 		self.batchnorm1 = nn.BatchNorm1d(num_features=512)
 		self.batchnorm2 = nn.BatchNorm1d(num_features=256)
 		self.batchnorm3 = nn.BatchNorm1d(num_features=128)

 		self.linear = nn.Linear(in_features=hidden_dim, out_features=num_classes)
 		self.relu = nn.ReLU()

 	def forward(self, x_batch):
 		padded_batch = pad_sequence(x_batch, batch_first=True)

 		embeds = self.word_embeddings(padded_batch)
 		embeds_t = embeds.transpose(1, 2)

 		cnn1 = self.relu(self.conv1(embeds_t))
 		cnn1 = self.batchnorm1(cnn1)

 		cnn2 = self.relu(self.conv2(cnn1))
 		cnn2 = self.batchnorm2(cnn2)

 		cnn3 = self.relu(self.conv3(cnn2))
 		cnn3 = self.batchnorm3(cnn3)

 		conv_outputs = []  # for interpretability
 		conv_outputs.append(cnn3)

 		rnn_input = cnn3.transpose(1, 2)

 		_, (lstm_h, _) = self.lstm(rnn_input)

 		linear_out = self.linear(lstm_h.squeeze())

 		return conv_outputs, linear_out



