from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn

# inherit from nn.module (it is a pytorch neural network module)
class LSTMActorCritic(nn.Module):
   
   # building the neural network parts
    def __init__(
        self,
        input_dim: int = 26,
        hidden_size: int = 128,
        num_layers: int = 1,
        action_dim: int = 7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
        )

        self.lstm_dropout = nn.Dropout(0.1) if num_layers > 1 else nn.Identity()

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    # hidden compressed memory
    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device = device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device = device)
        return (h,c)
    
    # forward pass
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = self.lstm_dropout(lstm_out)

        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        # Apply action mask: set invalid actions to -inf so softmax gives 0
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))

        return logits, values, new_hidden
    
