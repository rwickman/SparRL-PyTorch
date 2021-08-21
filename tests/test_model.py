import os, sys, unittest

file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)

from fake_args import FakeArgs
from model import *
from conf import *

class TestModel(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
        self.sparrl_net =  SparRLNet(self.args, 16).to(device)

    def test_forward_single(self):
        """Test running through a single full subgraph."""
        subgraph = torch.tensor([[1, 2, 3, 4, 3, 1]], device=device)
        local_stats = torch.tensor(
            [[[4,2],[4,1],[2,3], [2,3], [2,3], [2,3]]], device=device, dtype=torch.float32)
        global_stats = torch.tensor([[[5,25]]], device=device, dtype=torch.float32)

        q_vals = self.sparrl_net(subgraph, local_stats, global_stats)
        
        self.assertListEqual(list(q_vals.shape), [1,3,1])


        
