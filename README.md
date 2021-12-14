# SparRL-PyTorch

## To train a model on the karate graph over spearman:
```code
python3 code/main.py --edge_list graphs/karate.edgelist --episodes 500 --T_max 17 --save_dir karate_spearman_models/ --subgraph_len 32 --obj spearman --T_eval 32 --eval_batch_size 1 --decay_episodes 200
```

## Evalute over 8 episodes
```code
python3 code/main.py --edge_list graphs/karate.edgelist --episodes 8 --T_max 17 --save_dir karate_spearman_models/ --subgraph_len 32 --obj spearman --load --eval
```

## Show training graphs
```code
python3 code/util/plot_rewards.py --save_dir karate_spearman_models --reward_smooth_w 64
```


