import pandas as pd 
from matplotlib import pyplot as plt
import argparse

import os


figname = "World Wide - Gradient"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("output_folder")


    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)

    os.makedirs(args.output_folder, exist_ok=True)

    dictionary = {
        "s": "Susceptible",
        "i": "Infected",
        "r": "Recovered + Death",
    }

    scale = {
        "s": "1e8",
        "i": "1e5",
        "r": "1e5",
    }

    fig = plt.figure(0, figsize=(8, 12))
    fig.suptitle(figname, fontsize=16)
    for i, t in enumerate(["s", "i", "r"]):
        ax = fig.add_subplot(3, 1, i + 1)
        
        ax.plot(df["day"], df[t] / float(scale[t]), label=t + " predict")
        ax.plot(df["day"], df[t + "_gt"] / float(scale[t]), label=t + " ground truth")

        ax.set_xlabel("Day")
        ax.set_ylabel(dictionary[t] + f" ({scale[t]})")

        ax.legend()

        fig_path = os.path.join(args.output_folder, t + "_figure.png")

    fig.savefig(fig_path)

main()


