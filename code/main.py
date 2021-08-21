import argparse, os


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--episodes", type=int, default=2048,
            help="Number of episodes to train on.")

    graph_args = parser.add_argument_group("Graph")
    graph_args.add_argument("--edge_list",
            help="Filename of edge list of the graph.")
    graph_args.add_argument("--is_dir", action="store_true",
            help="Indicates the graph should be directed.")

    env_args = parser.add_argument_group("Environment")
    env_args.add_argument("--obj", default="pr",
            help="The minimization objective (spsp / pr).")
    env_args.add_argument("--subgraph_len", type=int, default=64,
        help="Size of the subgraphs.")

    net_args = parser.add_argument_group("SparRL Network")
    net_args.add_argument("--emb_size", type=int, default=256,
        help="Size of node and edge embeddings.")
    net_args.add_argument("--hidden_size", type=int, default=256,
        help="Number of hidden units in each FC layer for the SparRL network.")
    
    dqn_args = parser.add_argument_group("DQN")
    dqn_args.add_argument("--epsilon", type=float, default=0.95,
                    help="Initial epsilon used for epsilon-greedy in DQN.")
    dqn_args.add_argument("--min_epsilon", type=float, default=0.01,
                    help="Minimum epsilon value used for epsilon-greedy in DQN.")
    dqn_args.add_argument("--epsilon_decay", type=int, default=1024,
                    help="Epsilon decay step used for decaying the epsilon value in epsilon-greedy exploration.")
    dqn_args.add_argument("--dqn_steps", type=int, default=1,
                    help="Number of steps to use for multistep DQN.")
    dqn_args.add_argument("--tgt_tau", type=float, default=0.05,
                    help="The tau value to control the update rate of the target DQN parameters.")
    dqn_args.add_argument("--mem_cap", type=int, default=32768,
                    help="Replay memory capacity.")
    dqn_args.add_argument("--expert_mem_cap", type=int, default=16384,
                    help="Number of expert experiences in replay memory.")
    dqn_args.add_argument("--gamma", type=float, default=0.99,
                    help="Reward discount.")
    main(parser.parse_args())