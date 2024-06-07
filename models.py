import torch.nn as nn
import torch

class StockPredictor(nn.Module):
    def __init__(self, embedding_dim, price_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm_bert = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_price = nn.LSTM(price_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, bert_x, price_x):
        lstm_out_bert, _ = self.lstm_bert(bert_x)
        lstm_out_price, _ = self.lstm_price(price_x)
        # Only consider the output of the last LSTM cell
        last_out_bert = lstm_out_bert[:, -1, :]
        last_out_price = lstm_out_price[:, -1, :]
        # Concatenate the outputs of both LSTMs
        combined_out = torch.cat((last_out_bert, last_out_price), dim=1)
        output = self.head(combined_out)
        return output