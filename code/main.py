import argparse, os

from environment import Environment
from agents.rl_agent import RLAgent
from agents.expert_agent import ExpertAgent
from replay_memory import PrioritizedExReplay, ReplayMemory
from results_manager import ResultsManager
from graph import Graph
from agent_manager import AgentManager
from util import num_expert_episodes

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    memory = PrioritizedExReplay(args)
    graph = Graph(args)
    # print("graph.num_nodes", graph.num_nodes)

    if args.expert_episodes > 0:
        agent = RLAgent(args, memory, graph.num_nodes, ExpertAgent(args, graph))
    else:
        agent = RLAgent(args, memory, graph.num_nodes)
    

    env = Environment(args, agent, graph, "cuda")
    results_man = ResultsManager(args, agent, env)
    if args.eval:
        results_man.eval()
    else:
        
        # if args.expert_episodes > 0:
        #     expected_num_ep = num_expert_episodes(
        #         graph.get_num_edges(),
        #         args)
        #     print("expected_num_ep", expected_num_ep)
        #     args.expert_episodes = int(max(expected_num_ep * args.expert_edge_visits, args.workers))

        #     # Make it divisible by args.workers to make it simpler
        #     print("args.expert_episodes", args.expert_episodes)
        #     print("args.expert_episodes mod args.workers", args.expert_episodes % args.workers)
        #     args.expert_episodes += args.workers  - (args.expert_episodes % args.workers)
        #     print("args.expert_episodes", args.expert_episodes)
            

        # del graph
        agent_man = AgentManager(args, agent, results_man)
        agent_man.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2048,
            help="Number of episodes to train on.")
    parser.add_argument("--batch_size", type=int, default=64,
            help="Number of episodes to train on.")
    parser.add_argument("--save_iter", type=int, default=32,
            help="Number of episodes to wait till saving the model.")
    # parser.add_argument("--train_iter", type=int, default=3,
    #                 help="Number of gradient update steps after each episode.")
    parser.add_argument("--lr_warmup_steps", type=int, default=32,
                    help="Number of steps for linear LR warmup.")
    parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="Minimum learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=0.999,
                    help="Learning rate decay factor.")
    parser.add_argument("--no_lr_decay", action="store_true",
            help="Don't use LR decay.")
    parser.add_argument("--save_dir", default="models/",
            help="Models save directory..")
    parser.add_argument("--load", action="store_true",
            help="Load saved models.")
    parser.add_argument("--node_embs", default="",
            help="Pretrained node embeddings to load into the network.")
    parser.add_argument("--min_ep", type=int, default=256,
            help="Minimum number of episodes that have to be elapsed before training.")
    parser.add_argument("--warmup_eps", type=int, default=1,
            help="Minimum number of episodes that have to be elapsed before training.")
    parser.add_argument("--decay_episodes", type=int, default=10000,
            help="Number of episdoes elapsed before epsilon decays to minimum.")
    parser.add_argument("--reward_scaler_window", type=int, default=8192,
            help="Number of rewards save to compute statistics over rewards to standardize.")
    parser.add_argument("--com_labels", default="",
            help="True community labels.")
    parser.add_argument("--num_spsp_pairs", type=int, default=8192,
            help="Number of shortest path pairs.")
    parser.add_argument("--max_neighbors", type=int, default=64,
            help="Maximum number of neighbors to pay attention to.")


    graph_args = parser.add_argument_group("Graph")
    graph_args.add_argument("--edge_list", required=True,
            help="Filename of edge list of the graph.")
    graph_args.add_argument("--is_dir", action="store_true",
            help="Indicates the graph should be directed.")

    env_args = parser.add_argument_group("Environment")
    env_args.add_argument("--obj", default="spearman",
            help="The minimization objective (spsp / pr).")
    env_args.add_argument("--subgraph_len", type=int, default=64,
        help="Size of the subgraphs.")
    env_args.add_argument("--T_max", type=int, default=512,
        help="Maximum edges to prune.")
    env_args.add_argument("--T_lam", type=float, default=32.0,
        help="Lambda used for sampling number of edges to prune per episode from poisson distribution.")
    # env_args.add_argument("--T_n", type=float, default=512,
    #     help="n value used for sampling number of edges to prune per episode from beta-binomial distribution.")
    env_args.add_argument("--T_alpha", type=float, default=1.0,
        help="Alpha value used for sampling number of edges to prune per episode from beta-binomial distribution.")
    env_args.add_argument("--T_beta", type=float, default=3.0,
        help="Beta value used for sampling number of edges to prune per episode from beta-binomial distribution.")
    env_args.add_argument("--preprune_pct", type=float, default=0.9,
        help="Percentage of edges to preprune.")
    env_args.add_argument("--reward_factor", type=float, default=1.0,
        help="Reward scaling factor.")


    net_args = parser.add_argument_group("SparRL Network")
    net_args.add_argument("--emb_size", type=int, default=128,
        help="Size of node and edge embeddings.")
    net_args.add_argument("--hidden_size", type=int, default=128,
        help="Number of hidden units in each FC layer for the SparRL network.")
    net_args.add_argument("--drop_rate", type=float, default=0.1,
        help="Dropout rate.")
    net_args.add_argument("--num_enc_layers", type=int, default=1,
        help="Number of Transformer Encoder layers.")
    net_args.add_argument("--num_heads", type=int, default=4,
        help="Number attention heads.")
    net_args.add_argument("--dff", type=int, default=512,
        help="Number of units in the pointwise FFN .")

    dqn_args = parser.add_argument_group("DQN")
    dqn_args.add_argument("--epsilon", type=float, default=0.99,
                    help="Initial epsilon used for epsilon-greedy in DQN.")
    dqn_args.add_argument("--min_epsilon", type=float, default=0.1,
                    help="Minimum epsilon value used for epsilon-greedy in DQN.")
    dqn_args.add_argument("--epsilon_decay", type=int, default=1024,
                    help="Epsilon decay step used for decaying the epsilon value in epsilon-greedy exploration.")
    dqn_args.add_argument("--dqn_steps", type=int, default=1,
                    help="Number of steps to use for multistep DQN.")
    dqn_args.add_argument("--tgt_tau", type=float, default=0.005,
                    help="The tau value to control the update rate of the target DQN parameters.")
    dqn_args.add_argument("--mem_cap", type=int, default=32768,
                    help="Replay memory capacity.")
    dqn_args.add_argument("--expert_mem_cap", type=int, default=16384,
                    help="Number of expert experiences in replay memory.")
    dqn_args.add_argument("--gamma", type=float, default=0.95,
                    help="Reward discount.")
    dqn_args.add_argument("--max_grad_norm", type=float, default=2.0,
                    help="Maximum gradient norm.")
    dqn_args.add_argument("--eps", type=float, default=1e-9,
                    help="Epsilon used for preventing divide by zero errors.")
    dqn_args.add_argument("--per_alpha", type=float, default=0.6,
                    help="Alpha used for proportional priority.")
    dqn_args.add_argument("--per_beta", type=float, default=0.4,
                    help="Beta used for proportional priority.")
    dqn_args.add_argument("--noise_std", type=float, default=0.1,
                    help="Standard deviation of sampled sigma parameters that control noise.")

    il_args = parser.add_argument_group("Imitation Learning")
    il_args.add_argument("--expert_lam", type=float, default=0.001,
                    help="Weight of the expert margin classification loss.")
    il_args.add_argument("--expert_margin", type=float, default=0.001,
                    help="Margin value used for IL margin classification loss.") 
    il_args.add_argument("--expert_epsilon", type=float, default=0.0,
                    help="Epsilon value added to priority value when using PER.")
    il_args.add_argument("--expert_spar", default="eff",
                    help="Expert sparsification method.")
    il_args.add_argument("--expert_episodes", type=int, default=0.0,
                    help="Number of expert episodes.")
    il_args.add_argument("--expert_p", type=float, default=0.95,
                    help="Probability of observing all edges.")
    il_args.add_argument("--expert_edge_visits", type=float, default=1,
                    help="Number of times each edge should be expected to be visited during expert episodes.")

    ec_args = parser.add_argument_group("Expert Control")
    ec_args.add_argument("--ec_episodes", type=int, default=1,
                    help="Number of expert control episodes.")
    ec_args.add_argument("--T_ec", type=int, default=128,
                    help="Number edge to prune for expert control.")
    ec_args.add_argument("--ec_sig_val", type=float, default=0.1,
                    help="Expert control signddficant value threshold.")
    ec_args.add_argument("--no_ec", action="store_true",
            help="Don't use expert control.")

    eval_args = parser.add_argument_group("Evaluation")
    eval_args.add_argument("--T_eval", type=int, default=128,
            help="Number of edges to prune for evaluation during training.")
    eval_args.add_argument("--eval_iter", type=int, default=16,
            help="Number of episodes elapsed during training until another evaluation episode is ran.")
    eval_args.add_argument("--eval_batch_size", type=int, default=8,
            help="Batch size for evaluation.")



    eval_args.add_argument("--eval", action="store_true",
            help="Evaluate.")

    mp_args = parser.add_argument_group("Multiprocess")    
    mp_args.add_argument("--workers", type=int, default=1,
            help="Number of workers to use for training.")

    main(parser.parse_args())
