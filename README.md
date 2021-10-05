# SparRL-PyTorch

## To train a model on the karate graph over spearman:
```code
python3 code/main.py --edge_list graphs/karate/karate.edgelist --episodes 16000 --T_max 17 --save_dir karate_spearman_models/ --subgraph_len 32 --obj spearman
```

## Evalute over 8 episodes
```code
python3 code/main.py --edge_list graphs/karate/karate.edgelist --episodes 8 --T_max 17 --save_dir karate_spearman_models/ --subgraph_len 32 --obj spearman --load --eval
```

## Show training graphs
```code
python3 code/util/plot_rewards --save_dir karate_spearman_models --reward_smooth_w 64
```


