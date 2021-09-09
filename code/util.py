import numpy as np
from scipy.stats import betabinom

def num_expert_episodes(E, args):
    """Compute number of expert episodes.

    Args:
        E: number of edges
        s: maximum subgraph length
        p: probability of sampling each edge at least once.
        T_avg: average number of episodes.
    """
    T_avg = betabinom.stats(args.T_max, args.T_alpha, args.T_beta, moments="m")
    print("T_avg:", T_avg)  
    return np.log(1-args.expert_p) / (T_avg * np.log(1-args.subgraph_len/E))



