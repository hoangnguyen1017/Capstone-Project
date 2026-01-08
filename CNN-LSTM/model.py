import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):

    def __init__(
        self,
        input_dim=42,         
        cnn_channels=(64, 128),
        lstm_hidden=128,
        lstm_layers=1,
        num_classes=2
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)      
        x = self.cnn(x)            
        x = x.transpose(1, 2)      

        lstm_out, _ = self.lstm(x)
        feat = lstm_out[:, -1, :]  

        return self.fc(feat)
