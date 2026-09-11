import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_v1(nn.Module):
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(1, 12, kernel_size=2, stride=1)   # (B, 12, 12)
        self.bn   = nn.BatchNorm1d(12)
        self.pool = nn.MaxPool1d(kernel_size=2)                 # (B, 12, 6)

        # LSTM treats the post-pool axis as time (seq_len=6, feature=12)
        self.lstm = nn.LSTM(input_size=12, hidden_size=lstm_hidden,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):                 # x: (B, 1, 13)
        x = F.relu(self.bn(self.conv(x))) # (B, 12, 12)
        x = self.pool(x)                  # (B, 12, 6)
        x = x.transpose(1, 2)             # (B, 6, 12)  -- (batch, seq, feat)
        _, (h_n, _) = self.lstm(x)        # h_n: (1, B, 32)
        return self.fc(h_n.squeeze(0))    # (B, 8)

import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_v2(nn.Module):
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32):
        super().__init__()
        # CNN block -- two conv stages to match MATLAB's 15-layer count
        self.conv1 = nn.Conv1d(in_channels=1,  out_channels=12,
                               kernel_size=2, stride=1)   # (B, 12, 12)
        self.bn1   = nn.BatchNorm1d(12)

        self.conv2 = nn.Conv1d(in_channels=12, out_channels=12,
                               kernel_size=2, stride=1, padding=1)  # (B, 12, 13)
        self.bn2   = nn.BatchNorm1d(12)

        self.pool = nn.MaxPool1d(kernel_size=2)           # halves the length

        # LSTM over the post-pool axis as time
        self.lstm = nn.LSTM(input_size=12, hidden_size=lstm_hidden,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):                       # x: (B, 1, 13)
        x = F.relu(self.bn1(self.conv1(x)))     # (B, 12, 12)
        x = F.relu(self.bn2(self.conv2(x)))     # (B, 12, 13)  with padding=1
        x = self.pool(x)                        # (B, 12, 6)
        x = x.transpose(1, 2)                   # (B, 6, 12) -- (batch, seq, feat)
        _, (h_n, _) = self.lstm(x)              # h_n: (1, B, 32)
        return self.fc(h_n.squeeze(0))          # (B, 8) logits