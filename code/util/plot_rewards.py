import json
import argparse, os
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    def moving_average(x):
        return np.convolve(x, np.ones(args.reward_smooth_w), 'valid') / args.reward_smooth_w

    train_dict_file = os.path.join(args.save_dir, "train_dict.json")
    with open(train_dict_file) as f:
        train_dict = json.load(f)

    fig, axs = plt.subplots(2)
    axs[0].plot(moving_average(train_dict["avg_rewards"]))
    axs[1].plot(train_dict["mse_losses"])

    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="models",
        help="Models save directory.")
    parser.add_argument("--reward_smooth_w", type=int, default=8,
                    help="Window size for reward smoothing plot.")
    
    main(parser.parse_args())
