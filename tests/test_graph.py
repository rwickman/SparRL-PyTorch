import os, sys, unittest

file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../code")
sys.path.append(code_path)


from fake_args import FakeArgs
from conf import *
from graph import Graph


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.args = FakeArgs()
    
    def test_degree_directed(self):
        node_ids = [7,0,5]
        expected_degrees = [
            [1,1],
            [3, 0],
            [0, 3]
        ]
        
        self.args.is_dir = True

        g = Graph(self.args)
        
        degrees = g.degree(node_ids)

        # Check the in/out degrees for each node is correct 
        self.assertListEqual(
            expected_degrees,
            [list(d) for d in list(zip(degrees[0], degrees[1]))]) 

    def test_degree_undirected(self):
        node_ids = [7,0,5]
        expected_degrees = [2, 3, 3]

        g = Graph(self.args)
        degrees = g.degree(node_ids)

        # Check the degrees are correct        
        self.assertListEqual(expected_degrees, degrees)
