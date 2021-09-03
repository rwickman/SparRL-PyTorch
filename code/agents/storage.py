import torch
from dataclasses import dataclass

@dataclass
class State:
    subgraph: torch.Tensor
    global_stats: torch.Tensor
    local_stats: torch.Tensor
    mask: torch.Tensor = None


@dataclass
class Experience:
    state: State
    next_state: State
    action: int
    reward: float
    is_expert: bool = False
    gamma: float = 0.99


@dataclass
class StateMessage:
    """Message send between processes containing the state or previous experiences."""
    state: State = None
    mask: torch.Tensor = None
    ex_buffer: list = None


@dataclass
class ActionMessage:
    """Contains action given to child process."""
    action: int