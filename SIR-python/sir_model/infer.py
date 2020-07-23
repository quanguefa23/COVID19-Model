from models.sir import SirModel

import torch
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Training script for Sir Model")
    parser.add_argument("csv_path")
    parser.add_argument("population", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--start_day", default=30, type=int)
    parser.add_argument("--num_day", default=30, type=int)
    parser.add_argument("--output_file", default="output.csv")
    

    return parser.parse_args()


def infer(df, population, args):
    # Load model
    model = SirModel(args.beta, args.gamma)
    model.eval()

    fout = open(args.output_file, "w")
    fout.write("day,s,i,r,s_gt,i_gt,r_gt\n")

    # Load data
    r = df.death + df.recover
    r = np.array(r)
    i = np.array(df.confirmed) - r

    s = (-1) * (i + r - population)


    gt_s = s
    gt_i = i
    gt_r = r


    s = torch.tensor(s[0:1])
    i = torch.tensor(i[0:1])
    r = torch.tensor(r[0:1])


    infer_start = args.start_day + 1

    with torch.no_grad():
        for idx, day in enumerate(range(infer_start, infer_start + args.num_day)):
            s, i, r = model(s, i, r, population)
            s_gt = gt_s[idx+1]
            i_gt = gt_i[idx+1]
            r_gt = gt_r[idx+1]
            fout.write(f"{day},{int(s)},{int(i)},{int(r)},{s_gt},{i_gt},{r_gt}\n")


def load_df_and_population(args):
    df = pd.read_csv(args.csv_path)
    population = int(args.population)

    df = df[args.start_day:args.start_day+ args.num_day +4]

    return df, population


def main():
    args = parse_args()
    model = SirModel()

    df, population = load_df_and_population(args)

    infer(df, population, args)


if __name__ == "__main__":
    main()
