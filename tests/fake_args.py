

class FakeArgs:
    def __init__(self):
        self.emb_size = 8
        self.hidden_size = 8
        self.drop_rate = 0.0
        self.num_enc_layers = 1
        self.num_heads = 2
        self.dff = 16
        self.max_pos_enc = 100
        self.edge_list = "fake_graphs/fake_small_graph.txt"
        self.obj = ""
        self.is_dir = False