import pandas as pd
from matplotlib import pyplot as plt
import argparse

import os

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

    for t in ["s", "i", "r"]:
        fig = plt.Figure()
        ax = fig.add_subplot()
        
        ax.plot(df["day"], df[t], label=t + " predict")
        ax.plot(df["day"], df[t + "_gt"], label=t + " ground truth")

        ax.set_xlabel("Day")
        ax.set_ylabel(dictionary[t])

        ax.legend()

        fig_path = os.path.join(args.output_folder, t + "_figure.pdf")

        fig.savefig(fig_path)

main()


