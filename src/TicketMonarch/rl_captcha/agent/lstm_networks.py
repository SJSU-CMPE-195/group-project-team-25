from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# inherit from nn.module (it is a pytorch neural network module)
class LSTMActorCritic(nn.Module):
   
   # building the neural network parts
    def __init__(
        self,
        input_dim: int = 13,
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

        self.actor = nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64,1),
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        lstm_out, new_hidden = self.lstm(x,hidden)

        logits = self.actor(lstm_out)
        values = self.critic(lstm_out)

        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        return logits, values, new_hidden
    
    # softmax logits -> probablities + entropy + values
    def get_action_and_value(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        logits, values, new_hidden = self.forward(x, hidden)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = values.squeeze(-1)

        return action, log_prob, entropy, value, new_hidden
