import seaborn as sns
import json
import argparse, os
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="darkgrid")
def main(args):
    def moving_average(x):
        return np.convolve(x, np.ones(args.reward_smooth_w), 'valid') / args.reward_smooth_w

    train_dict_file = os.path.join(args.save_dir, "train_dict.json")
    with open(train_dict_file) as f:
        train_dict = json.load(f)

    man_dict_file = os.path.join(args.save_dir, "agent_man.json")
    with open(man_dict_file) as f:
        agent_man_dict = json.load(f)


    fig, axs = plt.subplots(3)
    axs[0].plot(moving_average(train_dict["avg_rewards"]))
    axs[0].set(ylabel="Avg. Episode Reward")
    axs[1].plot(moving_average(train_dict["mse_losses"]))
    axs[1].set(ylabel="MSE Loss")
    axs[2].plot(agent_man_dict["eval_rewards"])
    axs[2].set(ylabel="Final Reward")

    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="models",
        help="Models save directory.")
    parser.add_argument("--reward_smooth_w", type=int, default=8,
                    help="Window size for reward smoothing plot.")
    
    main(parser.parse_args())
