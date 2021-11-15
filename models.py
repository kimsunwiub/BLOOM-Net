import torch
import torch.nn as nn
from torch.nn import Parameter

# Model
def initialize_weights(network):
    """ Init weights with Xavier initialization """
    for name, param in network.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

class SpeechEnhancementModel(nn.Module):
    def __init__(self, hidden_size, num_layers, stft_features):
        super(SpeechEnhancementModel, self).__init__()
        self.rnn = nn.GRU(
            input_size=stft_features, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True)
        self.dnn = nn.Linear(hidden_size, stft_features)
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x):
        (batch_size, seq_len, num_features) = x.shape
        rnn_out, hn = self.rnn(x) 
        _, _, hidden_size = rnn_out.shape
        # x: (seq_len, batch, hidden_size)
        # hn: (num_layers, batch, hidden_size)
        
        rnn_out = rnn_out.reshape(batch_size*seq_len, hidden_size)
        rnn_out = self.dnn(rnn_out)
        rnn_out = self.sigmoid(rnn_out)
        rnn_out = rnn_out.reshape(batch_size, seq_len, num_features)
        return rnn_out