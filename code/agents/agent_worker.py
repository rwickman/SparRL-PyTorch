import torch
from dataclasses import dataclass

from agents.storage import StateMessage, State


class AgentWorker:
    """Agent that interacts with RLAgent in a different process."""
    def __init__(self, args, state_queue, action_queue):
        self.args = args
        self._state_queue = state_queue
        self._action_queue = action_queue
        self._ex_buffer = [] 

    def add_ex(self, ex):
        if not self.args.eval:
            self._ex_buffer.append(ex)

    def reset(self):
        """Reset for next episode."""
        # Send collected experiences
        self._state_queue.put(
            StateMessage(ex_buffer=self._ex_buffer))

        # Get empty to verify last state was received
        self._action_queue.get()

        self._ex_buffer = []


    def __call__(self, state: torch.Tensor, mask: torch.Tensor = None) -> int:
        # Send state to alllow for it to be processed by the model
        self._state_queue.put(
            StateMessage(state, mask))
        
        # Get action back
        action_msg = self._action_queue.get()
        
        return action_msg.action