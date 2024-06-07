import torch.nn as nn
import torch
import torch.optim as optim
import torchmetrics

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
    
    
class StockPredictorSent(nn.Module):
    def __init__(self, embedding_dim, price_dim, sent_dim, hidden_dim, num_layers):
        super(StockPredictorSent, self).__init__()
        self.lstm_bert = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_price = nn.LSTM(price_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_sent = nn.LSTM(sent_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, bert_x, price_x, sent_x):
        lstm_out_bert, _ = self.lstm_bert(bert_x)
        lstm_out_price, _ = self.lstm_price(price_x)
        lstm_out_sent, _ = self.lstm_sent(sent_x)
        # Only consider the output of the last LSTM cell
        last_out_bert = lstm_out_bert[:, -1, :]
        last_out_price = lstm_out_price[:, -1, :]
        last_out_sent = lstm_out_sent[:, -1, :]
        
        # Concatenate the outputs of both LSTMs
        combined_out = torch.cat((last_out_bert, last_out_price, last_out_sent), dim=1)
        output = self.head(combined_out)
        return output

class BertRnnTrainer:
    def __init__(self, model, dataloader_train, dataloader_test, num_epochs=10, learning_rate=0.001):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse = torchmetrics.MeanSquaredError()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_bert, batch_price, batch_y in self.dataloader_train:
                self.optimizer.zero_grad()
                outputs = self.model(batch_bert, batch_price.unsqueeze(-1))
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, MAE: {loss.item()**0.5}")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch_bert, batch_price, batch_y in self.dataloader_test:
                outputs = self.model(batch_bert, batch_price.unsqueeze(-1))
                self.mse(outputs.squeeze(), batch_y)
        print("Test MSE: ", self.mse.compute())
        print("Test MAE: ", self.mse.compute()**0.5)
        

class BertSentimentTrainer:
    def __init__(self, model, dataloader_train, dataloader_test, num_epochs=10, learning_rate=0.001):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse = torchmetrics.MeanSquaredError()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_bert, batch_price, batch_sent, batch_y in self.dataloader_train:
                self.optimizer.zero_grad()
                outputs = self.model(batch_bert, batch_price.unsqueeze(-1), batch_sent.unsqueeze(-1))
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, MAE: {loss.item()**0.5}")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch_bert, batch_price, batch_sent, batch_y in self.dataloader_test:
                outputs = self.model(batch_bert, batch_price.unsqueeze(-1), batch_sent.unsqueeze(-1))
                self.mse(outputs.squeeze(), batch_y)
        print("Test MSE: ", self.mse.compute())
        print("Test MAE: ", self.mse.compute()**0.5)