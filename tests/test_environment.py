import torch
import numpy as np
import os, sys, unittest

file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from fake_args import FakeArgs
from conf import *
from graph import Graph
from agents.rl_agent import RLAgent
from environment import Environment

class FakeAgent:
    def __init__(self):
        pass

    def __call__(self, subgraph):
        return 0


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.agent = FakeAgent()
        self.graph = Graph(self.args)
        self.env = Environment(self.args, self.agent, self.graph)
    
    def test_create_state(self):
        # Set fixed sample subgraph 
        fake_subgraph = [[0,9], [0,1], [0, 8], [9,5]]
        self.env.sample_subgraph = lambda x: fake_subgraph
        
        expected_local_stats = [[3,0], [2,0], [3,0], [2,0], [3,0], [2, 0], [2,0], [3,0]]
        expected_local_stats = torch.log(torch.tensor(expected_local_stats) + 1).tolist()

        expected_global_stats = [1.6094379124341003, 2.4849066497880004]

        state = self.env.create_state(4, 5, 1)

        # Verify the local statistics are correct
        self.assertListEqual(state.local_stats.squeeze(0).tolist(), expected_local_stats)
        
        # Verify the global statistics are correct
        global_stats = state.global_stats.squeeze(0).squeeze(0).tolist()
        self.assertAlmostEqual(global_stats[0], expected_global_stats[0])
        self.assertAlmostEqual(global_stats[1], expected_global_stats[1])

        # Verify the subgraph is correct
        expected_subgraph = (torch.tensor(fake_subgraph) + 1).flatten().tolist()
        self.assertListEqual(state.subgraph.squeeze(0).tolist(), expected_subgraph)
    
    def test_preprune_no_prune(self):
        """Test edge cases."""
        self.env.preprune(10)
        self.assertEqual(self.graph.get_num_edges(), 11)

        self.env.preprune(11)
        self.assertEqual(self.graph.get_num_edges(), 11)

    def test_preprune_prune(self):
        self.env.preprune(1)
        self.assertTrue(self.graph.get_num_edges() == 11 or self.graph.get_num_edges() == 10)

    def test_prune_edge(self):
        fake_subgraph = (torch.tensor([[0,9], [0,1], [0, 8], [9,5]], device=device) + 1).flatten().unsqueeze(0)
        self.env.prune_edge(0, fake_subgraph)
        self.assertIn((0,9), self.env._removed_edges)
        self.assertEqual(self.graph.get_num_edges(), 10)

        self.env.prune_edge(2, fake_subgraph)
        self.assertIn((0,8), self.env._removed_edges)
        self.assertEqual(self.graph.get_num_edges(), 9)

        self.env.prune_edge(1, fake_subgraph)
        self.assertIn((0,1), self.env._removed_edges)
        self.assertEqual(self.graph.get_num_edges(), 8)


