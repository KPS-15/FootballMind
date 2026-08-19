import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class FootballTemporalLSTM(nn.Module):
    """
    Deep Learning LSTM Model for temporal sequence feature encoding over 5-10 second frame sliding windows.
    """

    def __init__(self, input_size: int = 16, hidden_size: int = 64, num_layers: int = 2, num_actions: int = 8):
        super(FootballTemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_action = nn.Linear(hidden_size, num_actions)
        self.fc_receiver = nn.Linear(hidden_size, 11)  # 11 candidate teammates
        self.fc_danger = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        action_logits = self.fc_action(last_hidden)
        receiver_logits = self.fc_receiver(last_hidden)
        danger_logit = torch.sigmoid(self.fc_danger(last_hidden))

        return action_logits, receiver_logits, danger_logit
