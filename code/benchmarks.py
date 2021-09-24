import networkx as nx
import networkit as nk
import numpy as np
import argparse, json
from scipy import stats
from dataclasses import dataclass
from tqdm import tqdm
from collections import Counter
from typing import Dict

from graph import Graph
from reward_manager import RewardManager
from community_detection import CommunityDetection

@dataclass
class Metrics:
    ari: float
    spearman: float
    edges: int
    spsp_str: str
    spsp_freq: Dict[int, float]

LINE_LEN = 50

class Benchmarks:
    def __init__(self, args):
        self._args = args
        self._graph = Graph(self._args)
        if self._args.com_labels:
            self._com_detect = CommunityDetection(self._args, self._graph)
        self._org_pr = list(self._graph.get_page_ranks().values())
        self._reward_manager = RewardManager(self._args, self._graph)
        self._reward_manager.setup()

    def run(self):
        # Convert networkx graph to networkit graph
        G = nk.nxadapter.nx2nk(self._graph.get_G())
        G.indexEdges()

        metrics = {
            "LD": {},
            "EFF": {},
            "AD": {},
            "RE": {},
            "LS": {},
            "QS": {},
            "S" : {}
        }
        for key in metrics:
            metrics[key]["ARI"] = {}
            metrics[key]["Spearman"] = {}


        with open(self._args.output, "w") as f:
            for ratio in tqdm(self._args.ratios):
                # metrics[ratio] = {
                #     "ARI" : {},
                #     "Spearman" : {}
                #     }
                
                # Local Degree
                local_deg_spar = nk.sparsification.LocalDegreeSparsifier()
                local_deg_metrics = self.get_metrics(G, local_deg_spar, ratio)

                # Edge Forest Fire
                eff_spar = nk.sparsification.ForestFireSparsifier(0.6, 5.0)
                eff_metrics = self.get_metrics(G, eff_spar, ratio)
        
                # Algebraic Distance
                algebraic_dist_spar = nk.sparsification.AlgebraicDistanceSparsifier()
                algebraic_dist_metrics = self.get_metrics(G, algebraic_dist_spar, ratio)

                # Random Edge
                rand_spar = nk.sparsification.RandomEdgeSparsifier()
                rand_metrics = self.get_metrics(G, rand_spar, ratio)
                
                # Local Similarity
                local_sim_spar = nk.sparsification.LocalSimilaritySparsifier()
                local_sim_metrics = self.get_metrics(G, local_sim_spar, ratio)

                # Quadrilateral Simmelian
                quad_sim_spar = nk.sparsification.QuadrilateralSimmelianSparsifier()
                quad_sim_metrics = self.get_metrics(G, quad_sim_spar, ratio) 
                
                # Simmelian
                simmelian_spar = nk.sparsification.SimmelianSparsifierNonParametric()
                simmelian_metrics = self.get_metrics(G, simmelian_spar, ratio)

                # Write out results
                f.write("Ratio {:.0%}\n".format(ratio))
                f.write("-" * LINE_LEN)

                f.write("\n  ARI:\n")
                f.write("-" * (LINE_LEN//2))
                f.write("\n\tLocal Degree: {}\n".format(local_deg_metrics.ari))
                f.write("\tEdge Forest Fire: {}\n".format(eff_metrics.ari))
                f.write("\tAlgebraic Distance: {}\n".format(algebraic_dist_metrics.ari))
                f.write("\tRandom Edge: {}\n".format(rand_metrics.ari))
                f.write("\tLocal Similarity: {}\n".format(local_sim_metrics.ari))
                f.write("\tQuadrilateral Simmelian: {}\n".format(quad_sim_metrics.ari))
                f.write("\tSimmelian: {}\n\n".format(simmelian_metrics.ari))

                f.write("\n  Spearman's Rank Correlation Coefficient:\n")
                f.write("-" * (LINE_LEN//2))
                f.write("\n\tLocal Degree: {}\n".format(local_deg_metrics.spearman))
                f.write("\tEdge Forest Fire: {}\n".format(eff_metrics.spearman))
                f.write("\tAlgebraic Distance: {}\n".format(algebraic_dist_metrics.spearman))
                f.write("\tRandom Edge: {}\n".format(rand_metrics.spearman))
                f.write("\tLocal Similarity: {}\n".format(local_sim_metrics.spearman))
                f.write("\tQuadrilateral Simmelian: {}\n".format(quad_sim_metrics.spearman))
                f.write("\tSimmelian: {}\n\n".format(simmelian_metrics.spearman))

                f.write("\n  SPSP Deltas:\n")
                f.write("-" * (LINE_LEN//2))
                f.write("\n\tLocal Degree: {}\n".format(local_deg_metrics.spsp_str))
                f.write("\tEdge Forest Fire: {}\n".format(eff_metrics.spsp_str))
                f.write("\tAlgebraic Distance: {}\n".format(algebraic_dist_metrics.spsp_str))
                f.write("\tRandom Edge: {}\n".format(rand_metrics.spsp_str))
                f.write("\tLocal Similarity: {}\n".format(local_sim_metrics.spsp_str))
                f.write("\tQuadrilateral Simmelian: {}\n".format(quad_sim_metrics.spsp_str))
                f.write("\tSimmelian: {}\n\n".format(simmelian_metrics.spsp_str))

                f.write("\n  Edges:\n")
                f.write("-" * (LINE_LEN//2))
                f.write("\n\tLocal Degree: {}\n".format(local_deg_metrics.edges))
                f.write("\tEdge Forest Fire: {}\n".format(eff_metrics.edges))
                f.write("\tAlgebraic Distance: {}\n".format(algebraic_dist_metrics.edges))
                f.write("\tRandom Edge: {}\n".format(rand_metrics.edges))
                f.write("\tLocal Similarity: {}\n".format(local_sim_metrics.edges))
                f.write("\tQuadrilateral Simmelian: {}\n".format(quad_sim_metrics.edges))
                f.write("\tSimmelian: {}\n\n".format(simmelian_metrics.edges))


                # Save metrics
                metrics["EFF"]["ARI"][ratio] = eff_metrics.ari
                metrics["LD"]["ARI"][ratio] = local_deg_metrics.ari
                metrics["AD"]["ARI"][ratio] = algebraic_dist_metrics.ari
                metrics["RE"]["ARI"][ratio] = rand_metrics.ari
                metrics["LS"]["ARI"][ratio] = local_sim_metrics.ari
                metrics["QS"]["ARI"][ratio] = quad_sim_metrics.ari
                metrics["S"]["ARI"][ratio] = simmelian_metrics.ari
                
                metrics["EFF"]["Spearman"][ratio] = local_deg_metrics.spearman
                metrics["LD"]["Spearman"][ratio]= eff_metrics.spearman
                metrics["AD"]["Spearman"][ratio] = algebraic_dist_metrics.spearman
                metrics["RE"]["Spearman"][ratio] = rand_metrics.spearman
                metrics["LS"]["Spearman"][ratio] = local_sim_metrics.spearman
                metrics["QS"]["Spearman"][ratio] = quad_sim_metrics.spearman
                metrics["S"]["Spearman"][ratio] = simmelian_metrics.spearman

                # metrics["EFF"]["Spearman"][ratio] = local_deg_metrics.spsp_freq
                # metrics["LD"]["Spearman"][ratio]= eff_metrics.spsp_freq
                # metrics["AD"]["Spearman"][ratio] = algebraic_dist_metrics.spsp_freq
                # metrics["RE"]["Spearman"][ratio] = rand_metrics.spsp_freq
                # metrics["LS"]["Spearman"][ratio] = local_sim_metrics.spsp_freq
                # metrics["QS"]["Spearman"][ratio] = quad_sim_metrics.spsp_freq
                # metrics["S"]["Spearman"][ratio] = simmelian_metrics.spsp_freq


        with open(self._args.output.split(".")[0] + ".json", "w") as f:
            json.dump(metrics, f)




    def get_metrics(self, G, sparsifier, ratio):
        ari_arr = np.zeros(self._args.predict_runs)
        spearman_arr = np.zeros(self._args.predict_runs)
        edge_arr = np.zeros(self._args.predict_runs, dtype=np.int32)
        spsp_arr = np.zeros(self._args.predict_runs)
        spsp_freq = {}


        for i in range(self._args.predict_runs):
            # Reset the graph
            if i > 0:
                self._graph.replace_G(nk.nxadapter.nk2nx(G))
            
            self._reward_manager.setup()
            # Get Sparsified graph
            G_spar = sparsifier.getSparsifiedGraphOfSize(G, ratio)
            
            # Replace underlying graph
            self._graph.replace_G(nk.nxadapter.nk2nx(G_spar))
            
            # Compute Louvain ARI score
            if self._args.com_labels:
                self._com_detect._graph = self._graph
                ari_arr[i] = self._com_detect.ARI_louvain()
            else:
                ari_arr[i] = -1
            #print(self._reward_manager.compute_reward())
            if self._args.obj == "spsp":
                # spsp_diff = self._reward_manager.spsp_diff()
                # #print(spsp_diff)
                # spsp_counter = Counter(spsp_diff)
                # #print(spsp_counter)
                # spsp_freq = {} 
                # for k in spsp_counter:
                #     if k not in spsp_freq:
                #         spsp_freq[k] = 0
                #     spsp_freq[k] = spsp_counter[k]
                spsp_arr[i] = self._reward_manager.compute_reward()
            
            # Compute Spearman's cor coef for PageRank
            if self._args.obj == "spearman":
                updated_pr = list(self._graph.get_page_ranks().values())
                spearmanr = stats.spearmanr(self._org_pr, updated_pr)
                spearman_arr[i] = spearmanr.correlation
            
            edge_arr[i] = self._graph.get_num_edges()
            # spsp_str = "\n"
            # if self._args.obj == "spsp": 
            #     for k in spsp_freq:
            #         spsp_freq[k] /= len(spsp_diff)
            #     total_prob = 0
            #     for dist, prob in sorted(spsp_freq.items()):
            #         spsp_str += "\t  {}: {}\n".format(dist, prob)
            #         total_prob += prob
        spsp_str = "\n\tSPSP REWARD: " + str(spsp_arr.mean())
        # print("spsp_arr", spsp_arr)
        return Metrics(ari_arr.mean(), spearman_arr.mean(), edge_arr.mean(), spsp_str, spsp_freq)

def main(args):
    benchmarks = Benchmarks(args)
    benchmarks.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_list", required=True,
        help="Filename of edge list of the graph.")
    parser.add_argument("--com_labels", default=None,
            help="Community labels file..")
    parser.add_argument("--predict_runs", type=int, default=8,
        help="Number of times to run each benchmark approach.")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.8, 0.6, 0.4, 0.2],
        help="Ratio of edges to preserve when pruning.")
    parser.add_argument("-o", "--output", default="benchmarks.txt",
        help="Output file of benchmark results.")
    parser.add_argument("--is_dir", action="store_true",
            help="Use directed graph.")
    parser.add_argument("--obj", default="com",
            help="The minimization objective.")
    parser.add_argument("--save_dir", default="temp",
            help="Save directory.")
    parser.add_argument("--num_spsp_pairs", type=int, default=2048,
            help="Number of spsp pairs to sample per episode.")
    main(parser.parse_args())
