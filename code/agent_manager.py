import torch
# from multiprocessing import Process, Queue, set_start_method
import torch.multiprocessing as mp
from dataclasses import dataclass


from graph import Graph
from agents.agent_worker import AgentWorker
from environment import Environment
from replay_memory import PrioritizedExReplay
from agents.rl_agent import RLAgent
from agents.storage import ActionMessage, State
from conf import *
#from expert_control import ExpertControl

mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy('file_system')

@dataclass
class ChildProcess:
    process: mp.Process
    env: Environment
    state_queue: mp.Queue
    action_queue: mp.Queue

class AgentManager:
    """Manager multiple agent workers for multiprocess training."""
    def __init__(self, args, rl_agent: RLAgent):
        self.args = args
        self._rl_agent = rl_agent
        #self._expert_control = ExpertControl(self.args)
        self._init_workers()
    
    def _init_workers(self):
        """Initialize the child processes with duplicate environments."""
        self._child_processes = []
        for i in range(self.args.workers):
            # Stores the states from the AgentWorker
            state_queue = mp.Queue(1)

            # Stores the actions to the AgentWorker
            action_queue = mp.Queue(1)

            # Create worker to communcate to global model
            agent_worker = AgentWorker(self.args, state_queue, action_queue)

            # Create duplicate environment
            graph = Graph(self.args)
            env = Environment(self.args, agent_worker, graph)
            
            # Create the child process
            agent_process = mp.Process(target=env.run)
            agent_process.start()
            self._child_processes.append(
                ChildProcess(agent_process, env, state_queue, action_queue))

    def _terminate_episode(self, exs: list):
        """Add experiences from episode to replay buffer."""
        for ex in exs:
            # Move to CUDA device
            ex.state.subgraph = ex.state.subgraph.to(device)
            ex.state.local_stats = ex.state.local_stats.to(device)
            ex.state.global_stats = ex.state.global_stats.to(device)
            ex.state.mask = ex.state.mask.to(device)

            if ex.next_state:
                ex.next_state.subgraph = ex.next_state.subgraph.to(device)
                ex.next_state.local_stats = ex.next_state.local_stats.to(device)
                ex.next_state.global_stats = ex.next_state.global_stats.to(device)
                ex.next_state.mask = ex.next_state.mask.to(device)

            self._rl_agent.add_ex(ex)

        self._rl_agent.reset()

    def _aggregate_states(self, states: list) -> State:
        """Aggregate states from several workers.
        
        Returns:
            the aggregated state.
        """
        batch_size = len(states)
        subgraphs = torch.zeros(batch_size, self.args.subgraph_len * 2, device=device, dtype=torch.int32)
        global_stats = torch.zeros(batch_size, 1, NUM_GLOBAL_STATS, device=device)
        local_stats = torch.zeros(batch_size, self.args.subgraph_len * 2, NUM_LOCAL_STATS, device=device)
        masks = torch.zeros(batch_size, 1, 1, self.args.subgraph_len + 1, device=device)
        
        for i, state in enumerate(states):
            subgraphs[i, :state.subgraph.shape[1]] = state.subgraph
            global_stats[i] = state.global_stats
            local_stats[i, :state.local_stats.shape[1]] = state.local_stats
            masks[i] = state.mask
        
        return State(subgraphs, global_stats, local_stats, masks)

    def run(self):
        # Run several batch of episodes
        for e_i in range(self.args.episodes // self.args.workers):
            # Keep up with processes that have episodes still running
            nonterm_p_ids = list(range(len(self._child_processes)))
            
            # Run until all episodes are terminated
            while len(nonterm_p_ids) > 0:
                states = []

                # Get states from all agents
                for p_id in nonterm_p_ids.copy():

                    # Get the current state from this process
                    state_msg = self._child_processes[p_id].state_queue.get()
                    
                    # Check if episode has terminated
                    if state_msg.state is None:
                        # Remove the process ID
                        self._child_processes[p_id].action_queue.put(ActionMessage(-1))
                        nonterm_p_ids.remove(p_id)
                        self._terminate_episode(state_msg.ex_buffer)
                    else:
                        states.append(state_msg.state)

                # Predict on states
                if len(nonterm_p_ids) >= 1:
                    states = self._aggregate_states(states)
                    
                    # Get actions
                    with torch.no_grad():
                        self._rl_agent.reset_noise()
                        action = self._rl_agent(states)
                    
                    if len(nonterm_p_ids) == 1:
                        action = [action]

                    # Pass actions to child processes
                    for i, p_id in enumerate(nonterm_p_ids):
                        self._child_processes[p_id].action_queue.put(
                            ActionMessage(action[i]))
                    
            if self._rl_agent.is_ready_to_train:
                # Train the model
                for _ in range(self.args.train_iter):
                    self._rl_agent.reset_noise()
                    self._rl_agent.train()

            # Save the models
            print("SAVING")
            
            #print("Should Trigger EC:", self._expert_control._test_mean_reward(self._rl_agent))
            
            if e_i % 8 == 0:
                self._rl_agent.save()
            print("DONE SAVING")
        
        print("SAVING")
        self._rl_agent.save()
        print("DONE SAVING")